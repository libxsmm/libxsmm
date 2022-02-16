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
#include <libxsmm_intrinsics_x86.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

typedef enum my_eltwise_fuse {
  MY_ELTWISE_FUSE_NONE = 0,
  MY_ELTWISE_FUSE_BIAS = 1,
  MY_ELTWISE_FUSE_RELU = 2,
  /* 3 is reserved for tanh */
  MY_ELTWISE_FUSE_BIAS_RELU = 4,
  MY_ELTWISE_FUSE_RELU_WITH_MASK = 6,
  MY_ELTWISE_FUSE_BIAS_RELU_WITH_MASK = 7
} my_eltwise_fuse;

typedef enum my_pass {
  MY_PASS_FWD   = 1,
  MY_PASS_BWD_D = 2,
  MY_PASS_BWD_W = 4,
  MY_PASS_BWD   = 6
} my_pass;

typedef struct my_fc_fwd_config {
  libxsmm_blasint N;
  libxsmm_blasint C;
  libxsmm_blasint K;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint bk;
  libxsmm_blasint threads;
  my_eltwise_fuse fuse_type;
  libxsmm_blasint fwd_bf;
  libxsmm_blasint fwd_2d_blocking;
  libxsmm_blasint fwd_row_teams;
  libxsmm_blasint fwd_col_teams;
  /* TODO: add hyperpartitions support */
  libxsmm_blasint fwd_M_hyperpartitions;
  libxsmm_blasint fwd_N_hyperpartitions;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
  libxsmm_xmmfunction fwd_compute_kernel_strd_f32;
  libxsmm_xmmfunction fwd_compute_kernel2_strd_f32;
//  libxsmm_gemmfunction gemm_fwd;
//  libxsmm_gemmfunction gemm_fwd2;
//  libxsmm_smmfunction_reducebatch_strd gemm_fwd;
//  libxsmm_smmfunction_reducebatch_strd gemm_fwd2;
  libxsmm_meltwfunction_unary fwd_zero_kernel;
  libxsmm_meltwfunction_unary fwd_relu_kernel;
  libxsmm_meltwfunction_unary fwd_colbcast_copy_kernel;
} my_fc_fwd_config;

typedef struct my_fc_bwd_config {
  libxsmm_blasint N;
  libxsmm_blasint C;
  libxsmm_blasint K;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint bk;
  libxsmm_blasint threads;
  my_eltwise_fuse fuse_type;
  libxsmm_blasint bwd_bf;
  libxsmm_blasint bwd_2d_blocking;
  libxsmm_blasint bwd_col_teams;
  libxsmm_blasint bwd_row_teams;
  /* TODO: add hyperpartitions support */
  libxsmm_blasint bwd_M_hyperpartitions;
  libxsmm_blasint bwd_N_hyperpartitions;
  libxsmm_blasint upd_bf;
  libxsmm_blasint upd_2d_blocking;
  libxsmm_blasint upd_col_teams;
  libxsmm_blasint upd_row_teams;
  /* TODO: add hyperpartitions support */
  libxsmm_blasint upd_M_hyperpartitions;
  libxsmm_blasint upd_N_hyperpartitions;
  libxsmm_blasint ifm_subtasks;
  libxsmm_blasint ofm_subtasks;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
  libxsmm_xmmfunction bwd_compute_kernel_strd_f32;
  libxsmm_xmmfunction bwd_compute_kernel2_strd_f32;
//  libxsmm_smmfunction_reducebatch_strd gemm_bwd;
//  libxsmm_smmfunction_reducebatch_strd gemm_bwd2;
  libxsmm_xmmfunction upd_compute_kernel_strd_f32;
  libxsmm_xmmfunction upd_compute_kernel2_strd_f32;
//  libxsmm_smmfunction_reducebatch_strd gemm_upd;
//  libxsmm_smmfunction_reducebatch_strd gemm_upd2;
  libxsmm_meltwfunction_unary norm_to_normT_kernel;
  libxsmm_meltwfunction_unary bwd_relu_kernel;
  libxsmm_meltwfunction_unary bwd_zero_kernel;
  libxsmm_meltwfunction_unary upd_zero_kernel;
  libxsmm_meltwfunction_unary delbias_reduce_kernel;
} my_fc_bwd_config;

my_fc_fwd_config setup_my_fc_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bn,
                                 libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, my_eltwise_fuse fuse_type) {
  my_fc_fwd_config res;
  libxsmm_blasint lda = bk;
  libxsmm_blasint ldb = bc;
  libxsmm_blasint ldc = bk;
  libxsmm_blasint ld_zero = bk*bn;
  float alpha = 1.0f;
  float beta = 1.0f;
  float zerobeta = 0.0f;

  libxsmm_meltw_unary_shape  unary_shape;
  libxsmm_meltw_binary_shape binary_shape;

  libxsmm_bitfield  unary_flags;
  libxsmm_bitfield  binary_flags;

  libxsmm_gemm_shape l_shape;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  libxsmm_bitfield l_flags;
  libxsmm_bitfield l_prefetch_flags = 0;
  int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);

#ifdef USE_GEMM_EXT
  libxsmm_gemm_ext_unary_argops   l_argops;
  libxsmm_gemm_ext_binary_postops l_postops;

  memset( &l_argops,  0, sizeof(libxsmm_gemm_ext_unary_argops  ) );
  memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );
#endif

  libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32;

  /* setting up some handle values */
  res.N = N;
  res.C = C;
  res.K = K;
  res.bn = bn;
  res.bc = bc;
  res.bk = bk;
  res.threads = threads;
  res.fuse_type = fuse_type;

  /* setup parallelization strategy */
  res.fwd_M_hyperpartitions = 1; /* TODO: enable hyperpartitions */
  res.fwd_N_hyperpartitions = 1;
#if 0
  if (threads == 16) {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 2;
    res.fwd_row_teams = 8;
  } else
#endif

#if 0
  {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 1;
  }
#else
  res.fwd_bf = 1;
  res.fwd_2d_blocking = 0;
  res.fwd_col_teams = 1;
  res.fwd_row_teams = 1;

  if (res.C == 100 && res.K == 1024 && res.threads == 28) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 14;
    res.fwd_row_teams = 2;
  }

  if (res.C == 1024 && res.K == 1024 && res.threads == 28) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 7;
    res.fwd_row_teams = 4;
  }

  if (res.C == 100 && res.K == 1024 && res.threads == 40) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 10;
    res.fwd_row_teams = 4;
  }

  if (res.C == 1024 && res.K == 1024 && res.threads == 40) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 10;
    res.fwd_row_teams = 4;
  }

  if (res.C == 100 && res.K == 1024 && res.threads == 22) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 11;
    res.fwd_row_teams = 2;
  }

  if (res.C == 1024 && res.K == 1024 && res.threads == 22) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 11;
    res.fwd_row_teams = 2;
  }

  if (res.C == 100 && res.K == 1024 && res.threads == 64) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 8;
    res.fwd_row_teams = 8;
  }

  if (res.C == 1024 && res.K == 1024 && res.threads == 64) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 8;
    res.fwd_row_teams = 8;
  }

  if (res.C == 512 && res.K == 512 && res.threads == 28) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 1;
  }

  if (res.C == 1024 && res.K == 1 && res.threads == 28) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 1;
  }

  if (res.C == 512 && res.K == 512 && res.threads == 40) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 1;
  }

  if (res.C == 1024 && res.K == 1 && res.threads == 40) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 1;
  }

  if (res.C == 1024 && res.K == 1024 && res.threads == 20) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 5;
    res.fwd_row_teams = 4;
  }

  if (res.C == 100 && res.K == 1024 && res.threads == 20) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 5;
    res.fwd_row_teams = 4;
  }

  if (res.C == 1024 && res.K == 1024 && res.threads == 24) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 6;
    res.fwd_row_teams = 4;
  }

  if (res.C == 100 && res.K == 1024 && res.threads == 24) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 5;
    res.fwd_row_teams = 4;
  }

  if (res.C == 512 && res.K == 512 && res.threads == 24) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 5;
    res.fwd_row_teams = 4;
  }

  if (res.C == 512 && res.K == 512 && res.threads == 20) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 5;
    res.fwd_row_teams = 4;
  }

  if (res.C == 1024 && res.K == 1 && res.threads == 24) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 5;
    res.fwd_row_teams = 4;
  }

  if (res.C == 1024 && res.K == 1 && res.threads == 20) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 6;
    res.fwd_row_teams = 4;
  }
  if (res.C == 4096 && res.K == 4096 && res.threads == 8) {
    res.fwd_bf = 8/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 4;
    res.fwd_row_teams = 2;
  }
  if (res.C == 1024 && res.K == 1024 && res.threads == 8) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 4;
    res.fwd_row_teams = 2;
  }
#endif

  if ((res.C >= 512) && (res.threads == 22)) {
    res.fwd_bf = 4;
    while  ((res.C/res.bc) % res.fwd_bf != 0) {
      res.fwd_bf--;
    }
  }

#if 0
  res.fwd_bf = atoi(getenv("FWD_BF"));
  res.fwd_2d_blocking = atoi(getenv("FWD_2D_BLOCKING"));
  res.fwd_col_teams = atoi(getenv("FWD_COL_TEAMS"));
  res.fwd_row_teams = atoi(getenv("FWD_ROW_TEAMS"));
#endif

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */

  l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  //l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
  l_flags |= 0; // FIXME?
  l_prefetch_flags = prefetch_mode;

  libxsmm_blasint stride_a = res.bk*res.bc*sizeof(float);
  libxsmm_blasint stride_b = res.bc*res.bn*sizeof(float);

  l_shape                              = libxsmm_create_gemm_shape(res.bk, res.bn, res.bc, lda, ldb, ldc, dtype, dtype, dtype, dtype);
  l_brconfig                           = libxsmm_create_gemm_batch_reduce_config(LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, stride_a, stride_b, 0 /*br_unroll_hint*/);
  res.fwd_compute_kernel_strd_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
  if ( res.fwd_compute_kernel_strd_f32.gemm == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd failed. Bailing...!\n");
    exit(-1);
  }

#ifdef USE_GEMM_EXT
  if ((res.fuse_type & MY_ELTWISE_FUSE_BIAS) > 0) {
    l_postops.d_in_type      = dtype;
    l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
    l_postops.d_binary_type  = LIBXSMM_MELTW_TYPE_BINARY_ADD;
    l_postops.ldd            = NULL;
  }
  memset( &l_argops,  0, sizeof(libxsmm_gemm_ext_unary_argops  ) );
  memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );
  if ((res.fuse_type & MY_ELTWISE_FUSE_RELU) > 0) { // FIXME: TO be extended for relumask
    //l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
  }

  res.fwd_compute_kernel_strd_fused_f32.gemm_ext = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig,
      l_argops, l_postops );
  if (  res.fwd_compute_kernel_strd_fused_f32.gemm_ext == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_strd_fused_f32 failed. Bailing...!\n");
    exit(-1);
  }
#endif

/*
  res.gemm_fwd  = libxsmm_smmdispatch_reducebatch_strd(res.bk, res.bn, res.bc,
      res.bk*res.bc*sizeof(float), res.bc*res.bn*sizeof(float),
      &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
  if ( res.gemm_fwd == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd failed. Bailing...!\n");
    exit(-1);
  }
*/

  //l_flags |= ( zerobeta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
  l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  res.fwd_compute_kernel2_strd_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
  if ( res.fwd_compute_kernel2_strd_f32.gemm == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd2 failed. Bailing...!\n");
    exit(-1);
  }

#ifdef USE_GEMM_EXT
  memset( &l_argops,  0, sizeof(libxsmm_gemm_ext_unary_argops  ) );
  memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );
  if (res.fuse_type == MY_ELTWISE_FUSE_BIAS || res.fus& MY_ELTWISE_FUSE_BIAS) > 0) { // FIXME
    l_postops.d_in_type      = dtype;
    l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
    l_postops.d_binary_type  = LIBXSMM_MELTW_TYPE_BINARY_ADD;
    l_postops.ldd            = NULL;
  }
  if ((res.fuse_type & MY_ELTWISE_FUSE_RELU) > 0) { // FIXME: TO be extended for relumask
    //l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
  }

  res.fwd_compute_kernel2_strd_fused_f32.gemm_ext = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig,
      l_argops, l_postops );
  if (  res.fwd_compute_kernel2_strd_fused_f32.gemm_ext == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel2_strd_fused_f32 failed. Bailing...!\n");
    exit(-1);
  }
#endif

/*
  res.gemm_fwd2 = libxsmm_smmdispatch_reducebatch_strd(res.bk, res.bn, res.bc,
      res.bk*res.bc*sizeof(float), res.bc*res.bn*sizeof(float),
      &lda, &ldb, &ldc, &alpha, &zerobeta, NULL, NULL);
  if ( res.gemm_fwd2 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd2 failed. Bailing...!\n");
    exit(-1);
  }
*/


  /* Eltwise TPPs  */
  unary_shape           = libxsmm_create_meltw_unary_shape(res.bn * res.bk, 1, ld_zero, ld_zero, dtype, dtype, dtype);
  unary_flags           = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.fwd_zero_kernel   = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  //res.fwd_zero_kernel = libxsmm_dispatch_meltw_unary(bn*bk, 1, &ld_zero, &ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.fwd_zero_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  unary_shape         = libxsmm_create_meltw_unary_shape(res.bk, res.bn, ldc, ldc, dtype, dtype, dtype);
  if ( res.fuse_type == MY_ELTWISE_FUSE_RELU_WITH_MASK || res.fuse_type == MY_ELTWISE_FUSE_BIAS_RELU_WITH_MASK )
    unary_flags       = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  else
    unary_flags       = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.fwd_relu_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_RELU, unary_shape, unary_flags);
  //res.fwd_relu_kernel  = libxsmm_dispatch_meltw_unary(res.bk, res.bn, &ldc, &ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT, LIBXSMM_MELTW_TYPE_UNARY_RELU);
  if ( res.fwd_relu_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_relu_kernel failed. Bailing...!\n");
    exit(-1);
  }

  unary_shape                  = libxsmm_create_meltw_unary_shape(res.bk, res.bn, ldc, ldc, dtype, dtype, dtype);
  unary_flags                  = LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL;
  res.fwd_colbcast_copy_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, unary_flags);
  //res.fwd_colbcast_copy_kernel = libxsmm_dispatch_meltw_unary(bk, bn, &ldc, &ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY );
  if ( res.fwd_colbcast_copy_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_colbcast_fp32_copy_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* init scratch */
  res.scratch_size = 0;

  return res;
}

my_fc_bwd_config setup_my_fc_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bn,
    libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, my_eltwise_fuse fuse_type) {
  my_fc_bwd_config res;
  libxsmm_blasint lda = bc;
  libxsmm_blasint ldb = bk;
  libxsmm_blasint ldc = bc;
  libxsmm_blasint ld_zero_bwd = bc*bn;
  libxsmm_blasint ld_zero_upd = bk;
  libxsmm_blasint ld_relu_bwd = bk;
  libxsmm_blasint delbias_K = K;
  libxsmm_blasint delbias_N = N;
  float alpha = 1.0f;
  float beta = 1.0f;
  float zerobeta = 0.0f;
  int updflags = LIBXSMM_GEMM_FLAGS( 'N', 'T' );
  libxsmm_blasint updM;
  libxsmm_blasint updN;
  libxsmm_blasint ldaT = bk;

  libxsmm_meltw_unary_shape  unary_shape;
  libxsmm_meltw_binary_shape binary_shape;

  libxsmm_bitfield  unary_flags;
  libxsmm_bitfield  binary_flags;

  libxsmm_gemm_shape l_shape;
  libxsmm_gemm_batch_reduce_config l_brconfig;
  //libxsmm_gemm_ext_unary_argops l_argops;
  //libxsmm_gemm_ext_binary_postops l_postops;
  libxsmm_bitfield l_flags;
  libxsmm_bitfield l_prefetch_flags = 0;
  int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE); // FIXME?

  libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32;

  /* setting up some handle values */
  res.N = N;
  res.C = C;
  res.K = K;
  res.bn = bn;
  res.bc = bc;
  res.bk = bk;
  res.threads = threads;
  res.fuse_type = fuse_type;

  /* setup parallelization strategy */
  res.bwd_M_hyperpartitions = 1; /* TODO: enable hyperpartitions */
  res.upd_M_hyperpartitions = 1;
  res.bwd_N_hyperpartitions = 1;
  res.upd_N_hyperpartitions = 1;
#if 0
  if (threads == 16) {
    res.bwd_bf = 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 2;
    res.bwd_row_teams = 8;
    res.upd_bf = 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  } else
#endif
  res.bwd_bf = 1;
  res.bwd_2d_blocking = 0;
  res.bwd_col_teams = 1;
  res.bwd_row_teams = 1;
  res.upd_bf = 1;
  res.upd_2d_blocking = 0;
  res.upd_col_teams = 1;
  res.upd_row_teams = 1;
  res.ifm_subtasks = 1;
  res.ofm_subtasks = 1;


  if (res.C == 1024 && res.K == 1024 && res.threads == 22) {
    res.bwd_bf = ((res.K/res.bk) % 8 == 0) ? 8 : 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 11;
    res.bwd_row_teams = 2;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 7;
    res.upd_row_teams = 4;
    res.ifm_subtasks = 1/*((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if (res.C == 512 && res.K == 512 && res.threads == 22) {
    res.bwd_bf = ((res.K/res.bk) % 4 == 0) ? 4 : 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 11;
    res.bwd_row_teams = 2;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1/*((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if (res.C == 1024 && res.K == 1024 && res.threads == 64) {
    res.bwd_bf = ((res.K/res.bk) % 8 == 0) ? 8 : 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 8;
    res.bwd_row_teams = 8;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 7;
    res.upd_row_teams = 4;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if (res.C == 512 && res.K == 512 && res.threads == 64) {
    res.bwd_bf = ((res.K/res.bk) % 4 == 0) ? 4 : 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 8;
    res.bwd_row_teams = 8;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if (res.C == 100 && res.K == 1024 && res.threads == 28) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1/*((res.bc % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if (res.C == 1024 && res.K == 1024 && res.threads == 28) {
    res.bwd_bf = ((res.K/res.bk) % 8 == 0) ? 8 : 1;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 7;
    res.bwd_row_teams = 4;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 7;
    res.upd_row_teams = 4;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if (res.C == 512 && res.K == 512 && res.threads == 28) {
    res.bwd_bf = ((res.K/res.bk) % 4 == 0) ? 4 : 1;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if (res.C == 1024 && res.K == 1 && res.threads == 28) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 14;
    res.bwd_row_teams = 2;
    res.upd_bf = ((res.N/res.bn) % 2 == 0) ? 2 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if (res.C == 100 && res.K == 1024 && res.threads == 40) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = ((res.N/res.bn) % 10 == 0) ? 10 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1/*((res.bc % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if (res.C == 1024 && res.K == 1024 && res.threads == 40) {
    res.bwd_bf = ((res.K/res.bk) % 8 == 0) ? 8 : 1;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 7;
    res.bwd_row_teams = 4;
    res.upd_bf = ((res.N/res.bn) % 10 == 0) ? 10 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 7;
    res.upd_row_teams = 4;
    res.ifm_subtasks = 1/*((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if (res.C == 512 && res.K == 512 && res.threads == 40) {
    res.bwd_bf = ((res.K/res.bk) % 4 == 0) ? 4 : 1;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = ((res.N/res.bn) % 10 == 0) ? 10 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1/*((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if (res.C == 1024 && res.K == 1 && res.threads == 40) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 10;
    res.bwd_row_teams = 4;
    res.upd_bf = ((res.N/res.bn) % 2 == 0) ? 2 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1/*((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }


  if (res.C == 1024 && res.K == 1024 && res.threads == 20) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 5;
    res.bwd_row_teams = 4;
    res.upd_bf = ((res.N/res.bn) % 15 == 0) ? 15 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 5;
    res.upd_row_teams = 4;
    res.ifm_subtasks = 1/*((res.bc % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if (res.C == 100 && res.K == 1024 && res.threads == 20) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = ((res.N/res.bn) % 9 == 0) ? 9 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1/*((res.bc % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
    res.ofm_subtasks = ((res.bk % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
  }

  if (res.C == 1024 && res.K == 1024 && res.threads == 24) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 6;
    res.bwd_row_teams = 4;
    res.upd_bf = ((res.N/res.bn) % 15 == 0) ? 15 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 6;
    res.upd_row_teams = 4;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }
  if (res.C == 100 && res.K == 1024 && res.threads == 24) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 12;
    res.bwd_row_teams = 2;
    res.upd_bf = ((res.N/res.bn) % 15 == 0) ? 15 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 5;
    res.upd_row_teams = 4;
    res.ifm_subtasks = 1/*((res.bc % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }
  if (res.C == 512 && res.K == 512 && res.threads == 24) {
    res.bwd_bf = ((res.K/res.bk) % 4 == 0) ? 4 : 1;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 5;
    res.bwd_row_teams = 4;
    res.upd_bf = ((res.N/res.bn) % 15 == 0) ? 15 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 5;
    res.upd_row_teams = 4;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }
  if (res.C == 512 && res.K == 512 && res.threads == 20) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = ((res.N/res.bn) % 15 == 0) ? 15 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = ((res.bc % 4 == 0) && (res.upd_2d_blocking == 0)) ? 4 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }
  if (res.C == 1024 && res.K == 1 && res.threads == 24) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 5;
    res.bwd_row_teams = 4;
    res.upd_bf = 1/*((res.N/res.bn) % 1 == 0) ? 1 : 1*/;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 5;
    res.upd_row_teams = 4;
    res.ifm_subtasks = ((res.bc % 4 == 0) && (res.upd_2d_blocking == 0)) ? 4 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }
  if (res.C == 1024 && res.K == 1 && res.threads == 20) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 5;
    res.bwd_row_teams = 4;
    res.upd_bf = 1/*((res.N/res.bn) % 1 == 0) ? 1 : 1*/;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 6;
    res.upd_row_teams = 4;
    res.ifm_subtasks = 1/*((res.bc % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if ((res.K >= 512) && (res.threads == 22)) {
    res.bwd_bf = 4;
    while  ((res.K/res.bk) % res.bwd_bf != 0) {
      res.bwd_bf--;
    }
  }

  if ((res.N >= 512) && (res.threads == 22) && (res.ifm_subtasks == 1) && (res.ofm_subtasks == 1)) {
    res.upd_bf = 8;
    while  ((res.N/res.bn) % res.upd_bf != 0) {
      res.upd_bf--;
    }
  }

#if 0
  res.bwd_bf = atoi(getenv("BWD_BF"));
  res.bwd_2d_blocking = atoi(getenv("BWD_2D_BLOCKING"));
  res.bwd_col_teams = atoi(getenv("BWD_COL_TEAMS"));
  res.bwd_row_teams = atoi(getenv("BWD_ROW_TEAMS"));
  res.upd_bf = atoi(getenv("UPD_BF"));
  res.upd_2d_blocking = atoi(getenv("UPD_2D_BLOCKING"));
  res.upd_col_teams = atoi(getenv("UPD_COL_TEAMS"));
  res.upd_row_teams = atoi(getenv("UPD_ROW_TEAMS"));
  res.ifm_subtasks = atoi(getenv("IFM_SUBTASKS"));
  res.ofm_subtasks = atoi(getenv("OFM_SUBTASKS"));
#endif

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */

  l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  //l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
  l_flags |= 0; // FIXME?
  l_prefetch_flags = prefetch_mode;

  libxsmm_blasint stride_a = res.bk*res.bc*sizeof(float);
  libxsmm_blasint stride_b = res.bk*res.bn*sizeof(float);

  l_shape                              = libxsmm_create_gemm_shape(res.bc, res.bn, res.bk, lda, ldb, ldc, dtype, dtype, dtype, dtype);
  l_brconfig                           = libxsmm_create_gemm_batch_reduce_config(LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, stride_a, stride_b, 0 /*br_unroll_hint*/);
  res.bwd_compute_kernel_strd_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
  if ( res.bwd_compute_kernel_strd_f32.gemm == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd failed. Bailing...!\n");
    exit(-1);
  }

  /* BWD GEMM */
/*
  res.gemm_bwd  = libxsmm_smmdispatch_reducebatch_strd(res.bc, res.bn, res.bk,
      res.bk*res.bc*sizeof(float), res.bk*res.bn*sizeof(float),
      &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
  if ( res.gemm_bwd == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd failed. Bailing...!\n");
    exit(-1);
  }
*/

  //l_flags |= ( zerobeta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
  l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  res.bwd_compute_kernel2_strd_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
  if ( res.bwd_compute_kernel2_strd_f32.gemm == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd2 failed. Bailing...!\n");
    exit(-1);
  }

/*
  res.gemm_bwd2 = libxsmm_smmdispatch_reducebatch_strd(res.bc, res.bn, res.bk,
      res.bk*res.bc*sizeof(float), res.bk*res.bn*sizeof(float),
      &lda, &ldb, &ldc, &alpha, &zerobeta, NULL, NULL);
  if ( res.gemm_bwd2 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd2 failed. Bailing...!\n");
    exit(-1);
  }
*/

  unary_shape              = libxsmm_create_meltw_unary_shape(res.bk, res.bc, ldaT, lda, dtype, dtype, dtype);
  unary_flags              = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.norm_to_normT_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, unary_flags);
  //res.norm_to_normT_kernel = libxsmm_dispatch_meltw_unary(bk, bc, &ldaT, &lda, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
  if ( res.norm_to_normT_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP norm_to_normT_kernel failed. Bailing...!\n");
    exit(-1);
  }

  unary_shape         = libxsmm_create_meltw_unary_shape(res.bk, res.bn, ld_relu_bwd, ld_relu_bwd, dtype, dtype, dtype);
  if ( res.fuse_type == MY_ELTWISE_FUSE_RELU_WITH_MASK || res.fuse_type == MY_ELTWISE_FUSE_BIAS_RELU_WITH_MASK )
    unary_flags       = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  else
    unary_flags       = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.bwd_relu_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_RELU_INV, unary_shape, unary_flags);
  //res.bwd_relu_kernel = libxsmm_dispatch_meltw_unary(bk, bn, &ld_relu_bwd, &ld_relu_bwd, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT, LIBXSMM_MELTW_TYPE_UNARY_RELU_INV);
  if ( res.bwd_relu_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP bwd_relu_kernel failed. Bailing...!\n");
    exit(-1);
  }

  unary_shape         = libxsmm_create_meltw_unary_shape(res.bn * res.bc, 1, ld_zero_bwd, ld_zero_bwd, dtype, dtype, dtype);
  unary_flags         = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.bwd_zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  //res.bwd_zero_kernel = libxsmm_dispatch_meltw_unary(bn*bc, 1, &ld_zero_bwd, &ld_zero_bwd, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.bwd_zero_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP bwd_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* UPD GEMM */
  lda = res.bk;
  ldb = res.bc;
  ldc = res.bk;
  updM = res.bk/res.ofm_subtasks;
  updN = res.bc/res.ifm_subtasks;

  l_flags = LIBXSMM_GEMM_FLAGS('N', 'T');

  stride_a = res.K*res.bn*sizeof(float);
  stride_b = res.C*res.bn*sizeof(float);

  //l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
  l_flags |= 0; // FIXME?

  l_shape                              = libxsmm_create_gemm_shape(updM, updN, res.bn, lda, ldb, ldc, dtype, dtype, dtype, dtype);
  l_brconfig                           = libxsmm_create_gemm_batch_reduce_config(LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, stride_a, stride_b, 0 /*br_unroll_hint*/);
  res.upd_compute_kernel_strd_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
  if ( res.upd_compute_kernel_strd_f32.gemm == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_upd failed. Bailing...!\n");
    exit(-1);
  }
/*
  res.gemm_upd = libxsmm_smmdispatch_reducebatch_strd(updM, updN, res.bn,
      res.K*res.bn*sizeof(float), res.C*res.bn*sizeof(float),
      &lda, &ldb, &ldc, &alpha, &beta, &updflags, NULL);
  if ( res.gemm_upd == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_upd failed. Bailing...!\n");
    exit(-1);
  }
*/

  //l_flags |= ( zerobeta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
  l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;

  res.upd_compute_kernel2_strd_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
  if ( res.upd_compute_kernel2_strd_f32.gemm == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_upd2 failed. Bailing...!\n");
    exit(-1);
  }
/*
  res.gemm_upd2 = libxsmm_smmdispatch_reducebatch_strd(updM, updN, res.bn,
      res.K*res.bn*sizeof(float), res.C*res.bn*sizeof(float),
      &lda, &ldb, &ldc, &alpha, &zerobeta, &updflags, NULL);
  if ( res.gemm_upd2 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_upd2 failed. Bailing...!\n");
    exit(-1);
  }
*/

  /* JIT TPP kernels */
  unary_shape         = libxsmm_create_meltw_unary_shape(res.bk, res.bc, ld_zero_upd, ld_zero_upd, dtype, dtype, dtype);
  unary_flags         = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.upd_zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  //res.upd_zero_kernel = libxsmm_dispatch_meltw_unary(bk, bc, &ld_zero_upd, &ld_zero_upd, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.upd_zero_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP upd_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  unary_shape               = libxsmm_create_meltw_unary_shape(res.bk, res.bn, delbias_K, delbias_N, dtype, dtype, dtype);
  unary_flags               = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  res.delbias_reduce_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT, unary_shape, unary_flags);
  //res.delbias_reduce_kernel = libxsmm_dispatch_meltw_unary(bk, bn, &delbias_K, &delbias_N, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT);
  if ( res.delbias_reduce_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP delbias_reduce_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* init scratch */
  res.scratch_size =  sizeof(float) * ( (((size_t)res.C + (size_t)res.K) * (size_t)res.N) + ((size_t)res.C * (size_t)res.K) );


  return res;
}

void my_fc_fwd_exec( my_fc_fwd_config cfg, const float* wt_ptr, const float* in_act_ptr, float* out_act_ptr,
    const float* bias_ptr, unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch ) {
  const libxsmm_blasint nBlocksIFm = cfg.C / cfg.bc;
  const libxsmm_blasint nBlocksOFm = cfg.K / cfg.bk;
  const libxsmm_blasint nBlocksMB  = cfg.N / cfg.bn;

  libxsmm_gemm_param gemm_param;

  gemm_param.a.secondary = NULL;//(void*)cfg.A_offsets; // FIXME?
  gemm_param.b.secondary = NULL;//(void*)cfg.B_offsets; // FIXME?

#ifdef USE_GEMM_EXT
  libxsmm_gemm_param gemm_param_ext;
  gemm_param_ext.a.secondary = NULL;//(void*)cfg.A_offsets; // FIXME?
  gemm_param_ext.b.secondary = NULL;//(void*)cfg.B_offsets; // FIXME?
#endif

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
  /* number of tasks that could be run in parallel */
  const libxsmm_blasint work = nBlocksOFm * nBlocksMB;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % cfg.threads == 0) ?
    (work / cfg.threads) : ((work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* loop variables */
  libxsmm_blasint mb1ofm1 = 0, mb1 = 0, ofm1 = 0, ifm1 = 0;
  libxsmm_blasint N_tasks_per_thread = 0, M_tasks_per_thread = 0;
  libxsmm_blasint my_M_start = 0, my_M_end = 0, my_N_start = 0, my_N_end = 0;
  libxsmm_blasint my_col_id = 0, my_row_id = 0, col_teams = 0, row_teams = 0;

  LIBXSMM_VLA_DECL(4, float,           output, out_act_ptr, nBlocksOFm, cfg.bn, cfg.bk);
  LIBXSMM_VLA_DECL(4, const float,      input,  in_act_ptr, nBlocksIFm, cfg.bn, cfg.bc);
  LIBXSMM_VLA_DECL(4, const float,     filter,      wt_ptr, nBlocksIFm, cfg.bc, cfg.bk);
  LIBXSMM_VLA_DECL(2, const float,       bias,    bias_ptr,                     cfg.bk);
  LIBXSMM_VLA_DECL(4, unsigned char, relubitmask,    relu_ptr, nBlocksOFm, cfg.bn, cfg.bk/8);
  libxsmm_meltw_unary_param       eltwise_params;

  unsigned long long  blocks = nBlocksIFm;
  libxsmm_blasint CB_BLOCKS = nBlocksIFm, BF = 1;
  LIBXSMM_UNUSED( scratch );

  BF = cfg.fwd_bf;
  CB_BLOCKS = nBlocksIFm/BF;
  blocks = CB_BLOCKS;

  col_teams = cfg.fwd_col_teams;
  row_teams = cfg.fwd_row_teams;
  my_row_id = ltid % row_teams;
  my_col_id = ltid / row_teams;
  N_tasks_per_thread = LIBXSMM_UPDIV(nBlocksMB, col_teams);
  M_tasks_per_thread = LIBXSMM_UPDIV(nBlocksOFm, row_teams);
  my_N_start = LIBXSMM_MIN(my_col_id * N_tasks_per_thread, nBlocksMB);
  my_N_end = LIBXSMM_MIN((my_col_id+1) * N_tasks_per_thread, nBlocksMB);
  my_M_start = LIBXSMM_MIN(my_row_id * M_tasks_per_thread, nBlocksOFm);
  my_M_end = LIBXSMM_MIN((my_row_id+1) * M_tasks_per_thread, nBlocksOFm);

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  if (cfg.fwd_2d_blocking == 1) {
    if (BF > 1) {
      for (ifm1 = 0; ifm1 < BF; ++ifm1) {
        for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
          for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
#ifdef USE_GEMM_EXT
            if ( ifm1 == 0 ) {
              if ( cfg.fuse_type == MY_ELTWISE_FUSE_BIAS || cfg.fuse_type == MY_ELTWISE_FUSE_BIAS_RELU || cfg.fuse_type == MY_ELTWISE_FUSE_BIAS_RELU_WITH_MASK ) {
                  gemm_param_ext.op.tertiary = &n_blocks;
                  gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
                  gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
                  gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                  bias
                  cfg.fwd_compute_kernel2_strd_f32.gemm_ext( &gemm_param_ext ); /* beta = 0.0 + bias */
              } else { // since kernel.gemm and gemm_ext have beta = 1.0, init with zero
                eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                cfg.fwd_zero_kernel(&eltwise_params);
              }
            } else { /* not ifm1 = 0 */
              if ( ( ifm1 == BF-1 ) &&
                   (cfg.fuse_type == MY_ELTWISE_FUSE_RELU || cfg.fuse_type == MY_ELTWISE_FUSE_BIAS_RELU ||
                      cfg.fuse_type == MY_ELTWISE_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_ELTWISE_FUSE_BIAS_RELU_WITH_MASK)
                 ) {
                gemm_param_ext.op.tertiary = &n_blocks;
                gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
                gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
                gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                relumask
                cfg.fwd_compute_kernel_strd_f32.gemm_ext( &gemm_param_ext ); /* beta = 1.0 + relu */
              } else {
                gemm_param.op.tertiary = &blocks;
                gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
                gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
                gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                cfg.fwd_compute_kernel_strd_f32.gemm( &gemm_param ); /* beta = 1.0 */
              }
            }
#else /* for USE_GEMM_EXT */

            /* Initialize output slice */
            if ( ifm1 == 0 ) {
              if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
                eltwise_params.in.primary  = (void*) &LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk);
                eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                cfg.fwd_colbcast_copy_kernel(&eltwise_params);
              } else {
                eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                cfg.fwd_zero_kernel(&eltwise_params);
              }
            }

#if 1
            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            cfg.fwd_compute_kernel_strd_f32.gemm( &gemm_param );
#else
            /* BRGEMM */
            cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
                &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
#endif
            /* apply post BRGEMM fusion */
            if ( ifm1 == BF-1  ) {
              if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
                eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                eltwise_params.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
                cfg.fwd_relu_kernel(&eltwise_params);
              }
            }
#endif // for #ifdef USE_GEMM_EXT
          }
        }
      }
    } else {
      for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
        for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
#ifdef USE_GEMM_EXT
          if ( cfg.fuse_type == MY_ELTWISE_FUSE_BIAS || cfg.fuse_type == MY_ELTWISE_FUSE_BIAS_RELU || cfg.fuse_type == MY_ELTWISE_FUSE_BIAS_RELU_WITH_MASK ) {

            if ( cfg.fuse_type == MY_ELTWISE_FUSE_BIAS || cfg.fuse_type == MY_ELTWISE_FUSE_BIAS_RELU || cfg.fuse_type == MY_ELTWISE_FUSE_BIAS_RELU_WITH_MASK ) {
              gemm_param_ext.op.tertiary = &blocks;
              gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
              gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
              gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              bias
              cfg.fwd_compute_kernel2_strd_f32.gemm_ext( &gemm_param_ext ); /* beta = 0.0 + bias */
            } else {
              gemm_param_ext.op.tertiary = &blocks;
              gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
              gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
              gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              cfg.fwd_compute_kernel_strd_f32.gemm_ext( &gemm_param_ext ); /* beta = 0.0 + relu */
            }
          } else { /* no bias, no relu */
            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            cfg.fwd_compute_kernel2_strd_f32.gemm( &gemm_param ); /* beta = 0.0 */
          }

#else /* USE_GEMM_EXT */

          if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
            eltwise_params.in.primary  = (void*) &LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk);
            eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            cfg.fwd_colbcast_copy_kernel(&eltwise_params);
#if 1
            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            cfg.fwd_compute_kernel_strd_f32.gemm( &gemm_param );
#else
            cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
                &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
#endif
          } else {
#if 1
            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            cfg.fwd_compute_kernel2_strd_f32.gemm( &gemm_param );
#else
            cfg.gemm_fwd2( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
                &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
#endif
          }
          /* post GEMM fusion */
          if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
            eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            eltwise_params.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
            cfg.fwd_relu_kernel(&eltwise_params);
          }
#endif /* USE_GEMM_EXT */
        }
      }
    }
  } else {
    if (BF > 1) {
      for ( ifm1 = 0; ifm1 < BF; ++ifm1 ) {
        for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
          mb1  = mb1ofm1%nBlocksMB;
          ofm1 = mb1ofm1/nBlocksMB;
          /* Initialize output slice */
          if ( ifm1 == 0 ) {
            if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
              eltwise_params.in.primary  = (void*) &LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk);
              eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              cfg.fwd_colbcast_copy_kernel(&eltwise_params);
            } else {
              eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              cfg.fwd_zero_kernel(&eltwise_params);
            }
          }
#ifdef USE_GEMM_EXT
          if ( ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) && ( ifm1 == BF-1 )  ) {
            gemm_param_ext.op.tertiary = &blocks;
            gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
            gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            cfg.fwd_compute_kernel_strd_f32.gemm_ext( &gemm_param_ext );
          } else {
            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            cfg.fwd_compute_kernel_strd_f32.gemm( &gemm_param );
          }
#else /* USE_GEMM_EXT */

#if 1
          gemm_param.op.tertiary = &blocks;
          gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
          gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
          gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          cfg.fwd_compute_kernel_strd_f32.gemm( &gemm_param );
#else
          /* BRGEMM */
          cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);

          /* BRGEMM (from the previous place) */
          //cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
          //    &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
          //    &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
#endif
          /* post GEMM fusion */
          if ( ifm1 == BF-1  ) {
            if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
              eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              eltwise_params.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
              cfg.fwd_relu_kernel(&eltwise_params);
            }
          }
#endif /* USE_GEMM_EXT */
        }
      }
    } else {
      for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
        mb1  = mb1ofm1%nBlocksMB;
        ofm1 = mb1ofm1/nBlocksMB;

#ifdef USE_GEMM_EXT
        if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
          eltwise_params.in.primary  = (void*) &LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk);
          eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          cfg.fwd_colbcast_copy_kernel(&eltwise_params);

          gemm_param_ext.op.tertiary = &blocks;
          gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
          gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
          gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          cfg.fwd_compute_kernel_strd_f32.gemm_ext( &gemm_param_ext );
        } else {
          gemm_param_ext.op.tertiary = &blocks;
          gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
          gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
          gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          cfg.fwd_compute_kernel2_strd_f32.gemm_ext( &gemm_param_ext );
        }

#else /* USE_GEMM_EXT */
        if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
          eltwise_params.in.primary  = (void*) &LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk);
          eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          cfg.fwd_colbcast_copy_kernel(&eltwise_params);
#if 1
          gemm_param.op.tertiary = &blocks;
          gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
          gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
          gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          cfg.fwd_compute_kernel_strd_f32.gemm( &gemm_param );
#else
          cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
          // from the prevous place
          //cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
          //    &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
          //    &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
#endif
        } else {
#if 1
          gemm_param.op.tertiary = &blocks;
          gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
          gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
          gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          cfg.fwd_compute_kernel2_strd_f32.gemm( &gemm_param );
#else
          cfg.gemm_fwd2( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);

          // from the previous place
          //cfg.gemm_fwd2( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
          //    &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
          //    &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
#endif
        }
        /* post GEMM fusion */
        if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
          eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          eltwise_params.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
          cfg.fwd_relu_kernel(&eltwise_params);
        }
#endif /* USE_GEMM_EXT */
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

void my_fc_bwd_exec( my_fc_bwd_config cfg, const float* wt_ptr, float* din_act_ptr,
    const float* dout_act_ptr, float* dwt_ptr, const float* in_act_ptr,
    float* dbias_ptr, const unsigned char* relu_ptr, my_pass pass, int start_tid, int my_tid, void* scratch ) {
  /* here we assume that input and output blocking is similar */
  const libxsmm_blasint bn = cfg.bn;
  const libxsmm_blasint bk = cfg.bk;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint nBlocksIFm = cfg.C / bc;
  const libxsmm_blasint nBlocksOFm = cfg.K / bk;
  const libxsmm_blasint nBlocksMB  = cfg.N / bn;

  libxsmm_gemm_param gemm_param;

  gemm_param.a.secondary = NULL;//(void*)cfg.A_offsets; // FIXME?
  gemm_param.b.secondary = NULL;//(void*)cfg.B_offsets; // FIXME?

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks for transpose that could be run in parallel */
  const libxsmm_blasint eltwise_work = nBlocksOFm * nBlocksMB;
  /* compute chunk size */
  const libxsmm_blasint eltwise_chunksize = (eltwise_work % cfg.threads == 0) ? (eltwise_work / cfg.threads) : ((eltwise_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint eltwise_thr_begin = (ltid * eltwise_chunksize < eltwise_work) ? (ltid * eltwise_chunksize) : eltwise_work;
  const libxsmm_blasint eltwise_thr_end = ((ltid + 1) * eltwise_chunksize < eltwise_work) ? ((ltid + 1) * eltwise_chunksize) : eltwise_work;
  libxsmm_blasint mb1ofm1;

  /* number of tasks for transpose that could be run in parallel */
  const libxsmm_blasint dbias_work = nBlocksOFm;
  /* compute chunk size */
  const libxsmm_blasint dbias_chunksize = (dbias_work % cfg.threads == 0) ? (dbias_work / cfg.threads) : ((dbias_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint dbias_thr_begin = (ltid * dbias_chunksize < dbias_work) ? (ltid * dbias_chunksize) : dbias_work;
  const libxsmm_blasint dbias_thr_end = ((ltid + 1) * dbias_chunksize < dbias_work) ? ((ltid + 1) * dbias_chunksize) : dbias_work;

  /* loop variables */
  libxsmm_blasint ofm1 = 0, mb1 = 0, ofm2 = 0;

  float *grad_output_ptr = (((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) ? ((float*)scratch)+(cfg.C*cfg.K) : (float*)dout_act_ptr);
  LIBXSMM_VLA_DECL(4, const float, doutput_orig,    dout_act_ptr, nBlocksOFm, bn, bk);
  LIBXSMM_VLA_DECL(4,       float,      doutput, grad_output_ptr, nBlocksOFm, bn, bk);

  LIBXSMM_VLA_DECL(2,               float,    dbias, dbias_ptr,                     cfg.bk);
  LIBXSMM_VLA_DECL(4, const unsigned char, relubitmask,  relu_ptr, nBlocksOFm, cfg.bn, cfg.bk/8);
  libxsmm_meltw_unary_param eltwise_params;
  libxsmm_meltw_unary_param trans_param;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
    for ( mb1ofm1 = eltwise_thr_begin; mb1ofm1 < eltwise_thr_end; ++mb1ofm1 ) {
      mb1  = mb1ofm1%nBlocksMB;
      ofm1 = mb1ofm1/nBlocksMB;
      eltwise_params.in.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, doutput_orig, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      eltwise_params.out.primary  = &LIBXSMM_VLA_ACCESS(4, doutput, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      eltwise_params.in.secondary = (void*)&LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
      cfg.bwd_relu_kernel(&eltwise_params);
    }

    /* wait for eltwise to finish */
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
    for ( ofm1 = dbias_thr_begin; ofm1 < dbias_thr_end; ++ofm1 ) {
      eltwise_params.in.primary    = &LIBXSMM_VLA_ACCESS(4,  doutput, 0, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      eltwise_params.out.primary   = &LIBXSMM_VLA_ACCESS(2,  dbias, ofm1, 0, cfg.bk);
      cfg.delbias_reduce_kernel(&eltwise_params);
    }

    /* wait for eltwise to finish */
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if ( (pass & MY_PASS_BWD_D) == MY_PASS_BWD_D ) {
    const libxsmm_blasint use_2d_blocking = cfg.bwd_2d_blocking;

    /* number of tasks that could be run in parallel */
    const libxsmm_blasint work = nBlocksIFm * nBlocksMB;
    /* compute chunk size */
    const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

    /* number of tasks for transpose that could be run in parallel */
    const libxsmm_blasint transpose_work = nBlocksIFm * nBlocksOFm;
    /* compute chunk size */
    const libxsmm_blasint transpose_chunksize = (transpose_work % cfg.threads == 0) ? (transpose_work / cfg.threads) : ((transpose_work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
    const libxsmm_blasint transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

    /* loop variables */
    libxsmm_blasint ifm1 = 0, ifm1ofm1 = 0, mb1ifm1 = 0;
    libxsmm_blasint N_tasks_per_thread = 0, M_tasks_per_thread = 0, my_M_start = 0, my_M_end = 0, my_N_start = 0, my_N_end = 0, my_col_id = 0, my_row_id = 0, col_teams = 0, row_teams = 0;

    LIBXSMM_VLA_DECL(4, const float,    filter,          wt_ptr, nBlocksIFm, bc, bk);
    LIBXSMM_VLA_DECL(4,       float,    dinput,     din_act_ptr, nBlocksIFm, bn, bc);
    LIBXSMM_VLA_DECL(4,       float, filter_tr, (float*)scratch, nBlocksOFm, bk, bc);

    unsigned long long  blocks = nBlocksOFm;
    libxsmm_blasint KB_BLOCKS = nBlocksOFm, BF = 1;
    BF = cfg.bwd_bf;
    KB_BLOCKS = nBlocksOFm/BF;
    blocks = KB_BLOCKS;

    if (use_2d_blocking == 1) {
      col_teams = cfg.bwd_col_teams;
      row_teams = cfg.bwd_row_teams;
      my_row_id = ltid % row_teams;
      my_col_id = ltid / row_teams;
      N_tasks_per_thread = LIBXSMM_UPDIV(nBlocksMB, col_teams);
      M_tasks_per_thread = LIBXSMM_UPDIV(nBlocksIFm, row_teams);
      my_N_start = LIBXSMM_MIN(my_col_id * N_tasks_per_thread, nBlocksMB);
      my_N_end = LIBXSMM_MIN((my_col_id+1) * N_tasks_per_thread, nBlocksMB);
      my_M_start = LIBXSMM_MIN(my_row_id * M_tasks_per_thread, nBlocksIFm);
      my_M_end = LIBXSMM_MIN((my_row_id+1) * M_tasks_per_thread, nBlocksIFm);
    }

    /* transpose weight */
    for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
      ofm1 = ifm1ofm1 / nBlocksIFm;
      ifm1 = ifm1ofm1 % nBlocksIFm;
      trans_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4,    filter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
      trans_param.out.primary = &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1, 0, 0, nBlocksOFm, bk, bc);
      cfg.norm_to_normT_kernel(&trans_param);
    }

    /* wait for transpose to finish */
    libxsmm_barrier_wait(cfg.barrier, ltid);

    if (use_2d_blocking == 1) {
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for (ifm1 = my_M_start; ifm1 < my_M_end; ++ifm1) {
            for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
              /* Initialize intermediate f32 tensor */
              if ( ofm1 == 0 ) {
                eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dinput, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
                cfg.bwd_zero_kernel(&eltwise_params);
              }
#if 1
              gemm_param.op.tertiary = &blocks;
              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bk, bc );
              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk);
              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
              cfg.bwd_compute_kernel_strd_f32.gemm( &gemm_param );
#else
              /* BRGEMM */
              cfg.gemm_bwd( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bk, bc ),
                  &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
                  &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
#endif
            }
          }
        }
      } else {
        for (ifm1 = my_M_start; ifm1 < my_M_end; ++ifm1) {
          for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
#if 1
              gemm_param.op.tertiary = &blocks;
              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, 0, 0, 0, nBlocksOFm, bk, bc);
              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk);
              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
              cfg.bwd_compute_kernel2_strd_f32.gemm( &gemm_param );
#else
            /* BRGEMM */
            cfg.gemm_bwd2( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, 0, 0, 0, nBlocksOFm, bk, bc),
                &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
#endif
          }
        }
      }
    } else {
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
            mb1  = mb1ifm1%nBlocksMB;
            ifm1 = mb1ifm1/nBlocksMB;
            /* Initialize intermediate f32 tensor */
            if ( ofm1 == 0 ) {
              eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dinput, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
              cfg.bwd_zero_kernel(&eltwise_params);
            }
#if 1
            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bk, bc );
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
            cfg.bwd_compute_kernel_strd_f32.gemm( &gemm_param );
#else
            /* BRGEMM */
            cfg.gemm_bwd( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bk, bc ),
                &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);

            // from the previous case
            //cfg.gemm_bwd( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bk, bc ),
            //    &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
            //    &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
#endif
          }
        }
      } else {
        for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
          mb1  = mb1ifm1%nBlocksMB;
          ifm1 = mb1ifm1/nBlocksMB;
#if 1
            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, 0, 0, 0, nBlocksOFm, bk, bc);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
            cfg.bwd_compute_kernel2_strd_f32.gemm( &gemm_param );
#else
          /* BRGEMM */
          cfg.gemm_bwd2( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, 0, 0, 0, nBlocksOFm, bk, bc ),
              &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk),
              &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
          // from the previous case
          //cfg.gemm_bwd2( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, 0, 0, 0, nBlocksOFm, bk, bc),
          //    &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk),
          //    &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
#endif
        }
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if ( (pass & MY_PASS_BWD_W) == MY_PASS_BWD_W ) {
    /* number of tasks that could be run in parallel */
    const libxsmm_blasint ofm_subtasks = (cfg.upd_2d_blocking == 1) ? 1 : cfg.ofm_subtasks;
    const libxsmm_blasint ifm_subtasks = (cfg.upd_2d_blocking == 1) ? 1 : cfg.ifm_subtasks;
    const libxsmm_blasint bbk = (cfg.upd_2d_blocking == 1) ? bk : bk/ofm_subtasks;
    const libxsmm_blasint bbc = (cfg.upd_2d_blocking == 1) ? bc : bc/ifm_subtasks;
    const libxsmm_blasint work = nBlocksIFm * ifm_subtasks * nBlocksOFm * ofm_subtasks;
    const libxsmm_blasint Cck_work = nBlocksIFm * ifm_subtasks * ofm_subtasks;
    const libxsmm_blasint Cc_work = nBlocksIFm * ifm_subtasks;

    /* 2D blocking parameters  */
    libxsmm_blasint use_2d_blocking = cfg.upd_2d_blocking;
    libxsmm_blasint N_tasks_per_thread = 0, M_tasks_per_thread = 0, my_M_start = 0, my_M_end = 0, my_N_start = 0, my_N_end = 0, my_col_id = 0, my_row_id = 0, col_teams = 0, row_teams = 0;

    /* compute chunk size */
    const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
    libxsmm_blasint BF = cfg.upd_bf;

    /* loop variables */
    libxsmm_blasint ifm1ofm1 = 0, ifm1 = 0, ifm2 = 0, bfn = 0;

    /* Batch reduce related variables */
    unsigned long long  blocks = nBlocksMB/BF;

    LIBXSMM_VLA_DECL(4, const float,   input, in_act_ptr, nBlocksIFm, bn, bc);
    LIBXSMM_VLA_DECL(4,       float, dfilter,    dwt_ptr, nBlocksIFm, bc, bk);

    if (use_2d_blocking == 1) {
      col_teams = cfg.upd_col_teams;
      row_teams = cfg.upd_row_teams;
      my_row_id = ltid % row_teams;
      my_col_id = ltid / row_teams;
      N_tasks_per_thread = LIBXSMM_UPDIV(nBlocksIFm, col_teams);
      M_tasks_per_thread = LIBXSMM_UPDIV(nBlocksOFm, row_teams);
      my_N_start = LIBXSMM_MIN(my_col_id * N_tasks_per_thread, nBlocksIFm);
      my_N_end = LIBXSMM_MIN((my_col_id+1) * N_tasks_per_thread, nBlocksIFm);
      my_M_start = LIBXSMM_MIN(my_row_id * M_tasks_per_thread, nBlocksOFm);
      my_M_end = LIBXSMM_MIN((my_row_id+1) * M_tasks_per_thread, nBlocksOFm);
    }

    if (use_2d_blocking == 1) {
      if (BF == 1) {
        for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
          for (ifm1 = my_N_start; ifm1 < my_N_end; ++ifm1) {
#if 1
            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput, 0, ofm1, 0, 0, nBlocksOFm, bn, bk);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,   0, ifm1, 0, 0, nBlocksIFm, bn, bc);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
            cfg.upd_compute_kernel2_strd_f32.gemm( &gemm_param );
#else
            /* BRGEMM */
            cfg.gemm_upd2(&LIBXSMM_VLA_ACCESS(4, doutput, 0, ofm1, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, input,   0, ifm1, 0, 0, nBlocksIFm, bn, bc),
                &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk), &blocks);
#endif
          }
        }
      } else {
        for (bfn = 0; bfn < BF; bfn++) {
          for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
            for (ifm1 = my_N_start; ifm1 < my_N_end; ++ifm1) {
              /* initialize current work task to zero */
              if (bfn == 0) {
                eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
                cfg.upd_zero_kernel(&eltwise_params);
              }
#if 1
              gemm_param.op.tertiary = &blocks;
              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput, bfn*blocks, ofm1, 0, 0, nBlocksOFm, bn, bk);
              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,   bfn*blocks, ifm1, 0, 0, nBlocksIFm, bn, bc);
              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
              cfg.upd_compute_kernel_strd_f32.gemm( &gemm_param );
#else
              /* BRGEMM */
              cfg.gemm_upd( &LIBXSMM_VLA_ACCESS(4, doutput, bfn*blocks, ofm1, 0, 0, nBlocksOFm, bn, bk),
                  &LIBXSMM_VLA_ACCESS(4, input,   bfn*blocks, ifm1, 0, 0, nBlocksIFm, bn, bc),
                  &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk), &blocks);
#endif
            }
          }
        }
      }
    } else {
      if (BF == 1) {
        for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
          ofm1 = ifm1ofm1 / Cck_work;
          ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
          ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
          ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;

#if 1
          gemm_param.op.tertiary = &blocks;
          gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput, 0, ofm1, 0, ofm2*bbk, nBlocksOFm, bn, bk);
          gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,   0, ifm1, 0, ifm2*bbc, nBlocksIFm, bn, bc);
          gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk);
          cfg.upd_compute_kernel2_strd_f32.gemm( &gemm_param );
#else
          /* BRGEMM */
          cfg.gemm_upd2( &LIBXSMM_VLA_ACCESS(4, doutput, 0, ofm1, 0, ofm2*bbk, nBlocksOFm, bn, bk),
              &LIBXSMM_VLA_ACCESS(4, input,   0, ifm1, 0, ifm2*bbc, nBlocksIFm, bn, bc),
              &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
#endif
        }
      } else {
        for (bfn = 0; bfn < BF; bfn++) {
          for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
            ofm1 = ifm1ofm1 / Cck_work;
            ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
            ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
            ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;

            /* initialize current work task to zero */
            if (bfn == 0) {
              eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
              cfg.upd_zero_kernel(&eltwise_params);
            }

#if 1
            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput, bfn*blocks, ofm1, 0, ofm2*bbk, nBlocksOFm, bn, bk);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,   bfn*blocks, ifm1, 0, ifm2*bbc, nBlocksIFm, bn, bc);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk);
            cfg.upd_compute_kernel_strd_f32.gemm( &gemm_param );
#else
            /* BRGEMM */
            cfg.gemm_upd( &LIBXSMM_VLA_ACCESS(4, doutput, bfn*blocks, ofm1, 0, ofm2*bbk, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, input,   bfn*blocks, ifm1, 0, ifm2*bbc, nBlocksIFm, bn, bc),
                &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
#endif
          }
        }
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);
  }
}

