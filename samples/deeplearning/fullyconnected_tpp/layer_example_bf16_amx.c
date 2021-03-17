/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_sync.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

/* include c-based dnn library */
#include "../common/dnn_common.h"

typedef enum my_eltwise_fuse {
  MY_ELTWISE_FUSE_NONE = 0,
  MY_ELTWISE_FUSE_BIAS = 1,
  MY_ELTWISE_FUSE_RELU = 2,
  MY_ELTWISE_FUSE_BIAS_RELU = MY_ELTWISE_FUSE_BIAS | MY_ELTWISE_FUSE_RELU
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
  libxsmm_blasint fwd_column_teams;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
  libxsmm_bsmmfunction fwd_config_kernel;
  libxsmm_bsmmfunction tilerelease_kernel;
  libxsmm_bsmmfunction_reducebatch_strd gemm_fwd;
  libxsmm_bsmmfunction_reducebatch_strd gemm_fwd2;
  libxsmm_bmmfunction_reducebatch_strd gemm_fwd3;
  libxsmm_bmmfunction_reducebatch_strd_meltwfused gemm_fwd4;
  libxsmm_bmmfunction_reducebatch_strd_meltwfused gemm_fwd5;
  libxsmm_bmmfunction_reducebatch_strd_meltwfused gemm_fwd6;
  libxsmm_bmmfunction_reducebatch_strd_meltwfused gemm_fwd7;
  libxsmm_bmmfunction_reducebatch_strd_meltwfused gemm_fwd8;
  libxsmm_meltwfunction_unary fwd_cvtfp32bf16_kernel;
  libxsmm_meltwfunction_unary fwd_cvtfp32bf16_relu_kernel;
  libxsmm_meltwfunction_unary fwd_sigmoid_cvtfp32bf16_kernel;
  libxsmm_meltwfunction_unary fwd_zero_kernel;
  libxsmm_meltwfunction_unary fwd_copy_bf16fp32_kernel;
  libxsmm_meltwfunction_unary fwd_colbcast_bf16fp32_copy_kernel;
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
  libxsmm_blasint bwd_row_teams;
  libxsmm_blasint bwd_column_teams;
  libxsmm_blasint upd_bf;
  libxsmm_blasint upd_2d_blocking;
  libxsmm_blasint upd_row_teams;
  libxsmm_blasint upd_column_teams;
  libxsmm_blasint ifm_subtasks;
  libxsmm_blasint ofm_subtasks;
  size_t          scratch_size;
  size_t  doutput_scratch_mark;
  libxsmm_barrier* barrier;
  libxsmm_bsmmfunction bwd_config_kernel;
  libxsmm_bsmmfunction upd_config_kernel;
  libxsmm_bsmmfunction tilerelease_kernel;
  libxsmm_bsmmfunction_reducebatch_strd gemm_bwd;
  libxsmm_bsmmfunction_reducebatch_strd gemm_bwd2;
  libxsmm_bmmfunction_reducebatch_strd gemm_bwd3;
  libxsmm_bsmmfunction_reducebatch_strd gemm_upd;
  libxsmm_bsmmfunction_reducebatch_strd gemm_upd2;
  libxsmm_bmmfunction_reducebatch_strd gemm_upd3;
  libxsmm_meltwfunction_unary     bwd_cvtfp32bf16_kernel;
  libxsmm_meltwfunction_unary     upd_cvtfp32bf16_kernel;
  libxsmm_meltwfunction_unary            bwd_relu_kernel;
  libxsmm_meltwfunction_unary            bwd_zero_kernel;
  libxsmm_meltwfunction_unary            upd_zero_kernel;
  libxsmm_meltwfunction_unary           delbias_reduce_kernel;
  libxsmm_meltwfunction_transform       vnni_to_vnniT_kernel;
  libxsmm_meltwfunction_transform       norm_to_normT_kernel;
  libxsmm_meltwfunction_transform       norm_to_vnni_kernel;
  libxsmm_meltwfunction_transform       norm_to_vnni_kernel_wt;
} my_fc_bwd_config;

my_fc_fwd_config setup_my_fc_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bn,
                                 libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, my_eltwise_fuse fuse_type) {
  my_fc_fwd_config res;
  libxsmm_blasint lda = bk;
  libxsmm_blasint ldb = bc;
  libxsmm_blasint ldc = bk;
  libxsmm_blasint ld_zero = bk*bn;
  libxsmm_blasint ld_upconvert = K;
  float alpha = 1.0f;
  float beta = 1.0f;
  float zerobeta = 0.0f;
  libxsmm_meltw_flags fusion_flags;
  int l_flags, l_tc_flags;
  int l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
  libxsmm_blasint unroll_hint;

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
  if (threads == 16) {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 1;
    res.fwd_row_teams = 2;
    res.fwd_column_teams = 8;
  } else {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 0;
    res.fwd_row_teams = 1;
    res.fwd_column_teams = 1;
  }

#if 0
  res.fwd_bf = atoi(getenv("FWD_BF"));
  res.fwd_2d_blocking = atoi(getenv("FWD_2D_BLOCKING"));
  res.fwd_row_teams = atoi(getenv("FWD_ROW_TEAMS"));
  res.fwd_column_teams = atoi(getenv("FWD_COLUMN_TEAMS"));
#endif

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */
  l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
  unroll_hint = (res.C/res.bc)/res.fwd_bf;

  res.fwd_config_kernel = libxsmm_bsmmdispatch(res.bk, res.bn, res.bc, &lda, &ldb, &ldc, NULL, &beta, &l_tc_flags, NULL);
  if ( res.fwd_config_kernel == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP fwd_config_kernel failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_fwd = libxsmm_bsmmdispatch_reducebatch_strd_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &beta, &l_flags, NULL);
  if ( res.gemm_fwd == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_fwd2 = libxsmm_bsmmdispatch_reducebatch_strd_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL);
  if ( res.gemm_fwd2 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd2 failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_fwd3 = libxsmm_bmmdispatch_reducebatch_strd_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL);
  if ( res.gemm_fwd3 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd3 failed. Bailing...!\n");
    exit(-1);
  }
  fusion_flags = LIBXSMM_MELTW_FLAG_COLBIAS_OVERWRITE_C;
  res.gemm_fwd4 = libxsmm_bmmdispatch_reducebatch_strd_meltwfused_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL, LIBXSMM_MELTW_OPERATION_COLBIAS_ACT, LIBXSMM_DATATYPE_F32, fusion_flags, 0, 0, 0, 0);
  if ( res.gemm_fwd4 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd4 failed. Bailing...!\n");
    exit(-1);
  }
  fusion_flags = LIBXSMM_MELTW_FLAG_ACT_RELU_OVERWRITE_C;
  res.gemm_fwd5 = libxsmm_bmmdispatch_reducebatch_strd_meltwfused_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL, LIBXSMM_MELTW_OPERATION_COLBIAS_ACT, LIBXSMM_DATATYPE_F32, fusion_flags, 0, 0, 0, 0);
  if ( res.gemm_fwd5 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd5 failed. Bailing...!\n");
    exit(-1);
  }
  fusion_flags = LIBXSMM_MELTW_FLAG_ACT_SIGM_OVERWRITE_C;
  res.gemm_fwd6 = libxsmm_bmmdispatch_reducebatch_strd_meltwfused_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL, LIBXSMM_MELTW_OPERATION_COLBIAS_ACT, LIBXSMM_DATATYPE_F32, fusion_flags, 0, 0, 0, 0);
  if ( res.gemm_fwd6 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd6 failed. Bailing...!\n");
    exit(-1);
  }
  fusion_flags = LIBXSMM_MELTW_FLAG_COLBIAS_ACT_RELU_OVERWRITE_C;
  res.gemm_fwd7 = libxsmm_bmmdispatch_reducebatch_strd_meltwfused_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL, LIBXSMM_MELTW_OPERATION_COLBIAS_ACT, LIBXSMM_DATATYPE_F32, fusion_flags, 0, 0, 0, 0);
  if ( res.gemm_fwd7 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd7 failed. Bailing...!\n");
    exit(-1);
  }
  fusion_flags = LIBXSMM_MELTW_FLAG_COLBIAS_ACT_SIGM_OVERWRITE_C;
  res.gemm_fwd8 = libxsmm_bmmdispatch_reducebatch_strd_meltwfused_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL, LIBXSMM_MELTW_OPERATION_COLBIAS_ACT, LIBXSMM_DATATYPE_F32, fusion_flags, 0, 0, 0, 0);
  if ( res.gemm_fwd8 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd8 failed. Bailing...!\n");
    exit(-1);
  }

  /* Also JIT eltwise TPPs... */
  res.fwd_cvtfp32bf16_kernel = libxsmm_dispatch_meltw_unary(res.bk, res.bn, &ldc, &ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( res.fwd_cvtfp32bf16_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_cvtfp32bf16_kernel failed. Bailing...!\n");
    exit(-1);
  }
  res.fwd_cvtfp32bf16_relu_kernel = libxsmm_dispatch_meltw_unary(res.bk, res.bn, &ldc, &ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_BITMASK, LIBXSMM_MELTW_TYPE_UNARY_RELU);
  if ( res.fwd_cvtfp32bf16_relu_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_cvtfp32bf16_relu_kernel failed. Bailing...!\n");
    exit(-1);
  }
  res.fwd_sigmoid_cvtfp32bf16_kernel = libxsmm_dispatch_meltw_unary(res.bk, res.bn, &ldc, &ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_SIGMOID);
  if ( res.fwd_sigmoid_cvtfp32bf16_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_sigmoid_cvtfp32bf16_kernel failed. Bailing...!\n");
    exit(-1);
  }
  res.tilerelease_kernel = libxsmm_bsmmdispatch(res.bk, res.bk, res.bk, NULL, NULL, NULL, NULL, NULL, &l_tr_flags, NULL);
  if ( res.tilerelease_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP tilerelease_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.fwd_zero_kernel = libxsmm_dispatch_meltw_unary(bn*bk, 1, &ld_zero, &ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.fwd_zero_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.fwd_colbcast_bf16fp32_copy_kernel = libxsmm_dispatch_meltw_unary(bk, bn, &ldc, &ldc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( res.fwd_colbcast_bf16fp32_copy_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_colbcast_bf16fp32_copy_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.fwd_copy_bf16fp32_kernel = libxsmm_dispatch_meltw_unary(K, 1, &ld_upconvert, &ld_upconvert, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( res.fwd_copy_bf16fp32_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_copy_bf16fp32_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* init scratch */
  res.scratch_size = sizeof(float) *  LIBXSMM_MAX(res.K * res.N, res.threads * LIBXSMM_MAX(res.bk * res.bn, res.K));

  return res;
}

my_fc_bwd_config setup_my_fc_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bn,
                                 libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, my_eltwise_fuse fuse_type) {
  my_fc_bwd_config res;
  const libxsmm_trans_descriptor* tr_desc = 0;
  libxsmm_descriptor_blob blob;
  libxsmm_blasint lda = bk;
  libxsmm_blasint ldb = bc;
  libxsmm_blasint ldc = bk;
  libxsmm_blasint ld_zero_bwd = bc*bn;
  libxsmm_blasint ld_zero_upd = bk;
  libxsmm_blasint delbias_K = K;
  libxsmm_blasint delbias_N = N;
  float alpha = 1.0f;
  float beta = 1.0f;
  float zerobeta = 0.0f;
  libxsmm_blasint updM;
  libxsmm_blasint updN;
  libxsmm_meltw_flags fusion_flags;
  int l_flags, l_tc_flags;
  int l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
  libxsmm_blasint unroll_hint;
  size_t size_bwd_scratch;
  size_t size_upd_scratch;
  libxsmm_blasint bbk;
  libxsmm_blasint bbc;
  libxsmm_meltw_transform_flags trans_flags;
  libxsmm_blasint ldaT = bc;
  libxsmm_blasint ldb_orig= bc;

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
  if (threads == 16) {
    res.bwd_bf = 1;
    res.bwd_2d_blocking = 1;
    res.bwd_row_teams = 2;
    res.bwd_column_teams = 8;
    res.upd_bf = 1;
    res.upd_2d_blocking = 0;
    res.upd_row_teams = 1;
    res.upd_column_teams = 1;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  } else {
    res.bwd_bf = 1;
    res.bwd_2d_blocking = 0;
    res.bwd_row_teams = 1;
    res.bwd_column_teams = 1;
    res.upd_bf = 1;
    res.upd_2d_blocking = 0;
    res.upd_row_teams = 1;
    res.upd_column_teams = 1;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  }

  bbk = (res.upd_2d_blocking == 1) ? bk : bk/res.ofm_subtasks;
  bbc = (res.upd_2d_blocking == 1) ? bc : bc/res.ifm_subtasks;

#if 0
  res.bwd_bf = atoi(getenv("BWD_BF"));
  res.bwd_2d_blocking = atoi(getenv("BWD_2D_BLOCKING"));
  res.bwd_row_teams = atoi(getenv("BWD_ROW_TEAMS"));
  res.bwd_column_teams = atoi(getenv("BWD_COLUMN_TEAMS"));
  res.upd_bf = atoi(getenv("UPD_BF"));
  res.upd_2d_blocking = atoi(getenv("UPD_2D_BLOCKING"));
  res.upd_row_teams = atoi(getenv("UPD_ROW_TEAMS"));
  res.upd_column_teams = atoi(getenv("UPD_COLUMN_TEAMS"));
  res.ifm_subtasks = atoi(getenv("IFM_SUBTASKS"));
  res.ofm_subtasks = atoi(getenv("OFM_SUBTASKS"));
#endif

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */
  /* BWD GEMM */
  l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
  unroll_hint = (res.K/res.bk)/res.bwd_bf;

  res.gemm_bwd = libxsmm_bsmmdispatch_reducebatch_strd_unroll(res.bc, res.bn, res.bk, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bk*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &ldb, &lda, &ldb, &alpha, &beta, &l_flags, NULL);
  if ( res.gemm_bwd == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_bwd2 = libxsmm_bsmmdispatch_reducebatch_strd_unroll(res.bc, res.bn, res.bk, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bk*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &ldb, &lda, &ldb, &alpha, &zerobeta, &l_flags, NULL);
  if ( res.gemm_bwd2 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd2 failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_bwd3 = libxsmm_bmmdispatch_reducebatch_strd_unroll(res.bc, res.bn, res.bk, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bk*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &ldb, &lda, &ldb, &alpha, &zerobeta, &l_flags, NULL);
  if ( res.gemm_bwd3 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd3 failed. Bailing...!\n");
    exit(-1);
  }
  res.bwd_config_kernel = libxsmm_bsmmdispatch(res.bc, res.bn, res.bk, &ldb, &lda, &ldb, NULL, &beta, &l_tc_flags, NULL);
  if ( res.bwd_config_kernel == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP bwd_config_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* Also JIT eltwise TPPs... */
  res.bwd_cvtfp32bf16_kernel  = libxsmm_dispatch_meltw_unary(res.bc, res.bn, &ldb, &ldb, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( res.bwd_cvtfp32bf16_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP bwd_cvtfp32bf16_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.bwd_relu_kernel  = libxsmm_dispatch_meltw_unary(res.bc, res.bn, &ldb, &ldb, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_BITMASK, LIBXSMM_MELTW_TYPE_UNARY_RELU_INV);
  if ( res.bwd_relu_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP bwd_relu_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.bwd_zero_kernel = libxsmm_dispatch_meltw_unary(bn*bc, 1, &ld_zero_bwd, &ld_zero_bwd, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.bwd_zero_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP bwd_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* JITing the tranpose kernel */
  trans_flags = LIBXSMM_MELTW_FLAG_TRANSFORM_VNNI_TO_VNNIT;
  res.vnni_to_vnniT_kernel = libxsmm_dispatch_meltw_transform(bk, bc, &lda, &ldaT, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, trans_flags);
  if ( res.vnni_to_vnniT_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP vnni_to_vnniT_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* UPD GEMM */
  lda = res.bk;
  ldb = res.bn;
  ldc = res.bk;
  updM = res.bk/res.ofm_subtasks;
  updN = res.bc/res.ifm_subtasks;

  l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
  unroll_hint = (res.N/res.bn)/res.upd_bf;
  res.gemm_upd = libxsmm_bsmmdispatch_reducebatch_strd_unroll(updM, updN, res.bn, res.bk*res.bn*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &beta, &l_flags, NULL);
  if ( res.gemm_upd == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_upd failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_upd2 = libxsmm_bsmmdispatch_reducebatch_strd_unroll(updM, updN, res.bn, res.bk*res.bn*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL);
  if ( res.gemm_upd2 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_upd2 failed. Bailing...!\n");
    exit(-1);
  }
  l_flags = l_flags | LIBXSMM_GEMM_FLAG_VNNI_C;
  res.gemm_upd3 = libxsmm_bmmdispatch_reducebatch_strd_unroll(updM, updN, res.bn, res.bk*res.bn*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL);
  if ( res.gemm_upd3 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_upd3 failed. Bailing...!\n");
    exit(-1);
  }
  res.upd_config_kernel = libxsmm_bsmmdispatch(updM, updN, res.bn, &lda, &ldb, &ldc, NULL, &beta, &l_tc_flags, NULL);
  if ( res.upd_config_kernel == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP upd_config_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.tilerelease_kernel = libxsmm_bsmmdispatch(res.bk, res.bk, res.bk, NULL, NULL, NULL, NULL, NULL, &l_tr_flags, NULL);
  if ( res.tilerelease_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP tilerelease_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* Also JIT eltwise TPPs... */
  res.upd_cvtfp32bf16_kernel  = libxsmm_dispatch_meltw_unary(bbk, bbc, &ldc, &ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( res.upd_cvtfp32bf16_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP upd_cvtfp32bf16_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.upd_zero_kernel = libxsmm_dispatch_meltw_unary(bbk, bbc, &ld_zero_upd, &ld_zero_upd, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.upd_zero_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP upd_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.delbias_reduce_kernel = libxsmm_dispatch_meltw_unary(bk, bn, &delbias_K, &delbias_N, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT);
  if ( res.delbias_reduce_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP delbias_reduce_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* JITing the tranpose kernels */
  trans_flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI;
  res.norm_to_vnni_kernel = libxsmm_dispatch_meltw_transform(bk, bn, &lda, &lda, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, trans_flags);
  if ( res.norm_to_vnni_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP norm_to_vnni_kernel failed. Bailing...!\n");
    exit(-1);
  }

  trans_flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI;
  res.norm_to_vnni_kernel_wt = libxsmm_dispatch_meltw_transform(bbk, bbc, &ldc, &ldc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, trans_flags);
  if ( res.norm_to_vnni_kernel_wt == NULL ) {
    fprintf( stderr, "JIT for TPP norm_to_vnni_kernel failed. Bailing...!\n");
    exit(-1);
  }

  trans_flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_NORMT;
  res.norm_to_normT_kernel = libxsmm_dispatch_meltw_transform(bc, bn, &ldb_orig, &ldb, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, trans_flags);
  if ( res.norm_to_normT_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP norm_to_normT_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* init scratch */
  size_bwd_scratch = sizeof(float) * LIBXSMM_MAX(res.C * res.N, res.threads * res.bc * res.bn) + sizeof(libxsmm_bfloat16) * res.C * res.K;
  size_upd_scratch = sizeof(float) * LIBXSMM_MAX(res.C * res.K, res.threads * res.bc * res.bk) + sizeof(libxsmm_bfloat16) * res.threads * res.bk * res.bc + sizeof(libxsmm_bfloat16) * (res.N * (res.C + res.K));
  res.scratch_size = LIBXSMM_MAX(size_bwd_scratch, size_upd_scratch) + 2 * sizeof(libxsmm_bfloat16) * res.N * res.K;
  res.doutput_scratch_mark = LIBXSMM_MAX(size_bwd_scratch, size_upd_scratch) ;

  return res;
}

void my_fc_fwd_exec( my_fc_fwd_config cfg, const libxsmm_bfloat16* wt_ptr, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr,
                     const libxsmm_bfloat16* bias_ptr, unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch ) {
  const libxsmm_blasint nBlocksIFm = cfg.C / cfg.bc;
  const libxsmm_blasint nBlocksOFm = cfg.K / cfg.bk;
  const libxsmm_blasint nBlocksMB  = cfg.N / cfg.bn;
  const libxsmm_blasint bn = cfg.bn;
  const libxsmm_blasint bk = cfg.bk;
  const libxsmm_blasint lpb = 2;
  const libxsmm_blasint bc_lp = cfg.bc/lpb;
  /* const libxsmm_blasint bc = cfg.bc;*/
  libxsmm_blasint use_2d_blocking = cfg.fwd_2d_blocking;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
  /* number of tasks that could be run in parallel */
  const libxsmm_blasint work = nBlocksOFm * nBlocksMB;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* loop variables */
  libxsmm_blasint mb1ofm1 = 0, mb1 = 0, ofm1 = 0, ifm1 = 0;
  libxsmm_blasint im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16,       output,  out_act_ptr, nBlocksOFm, cfg.bn, cfg.bk);
  LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16,  input,   in_act_ptr,  nBlocksIFm, cfg.bn, cfg.bc);
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, filter,       wt_ptr, nBlocksIFm, bc_lp, cfg.bk, lpb);
  LIBXSMM_VLA_DECL(4, float, output_f32, (float*)scratch, nBlocksOFm, bn, bk);
  libxsmm_meltw_gemm_param gemm_eltwise_params;
  libxsmm_blasint mb2 = 0;
  float* fp32_bias_scratch =  ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) ? (float*)scratch + ltid * cfg.K : NULL;
  LIBXSMM_VLA_DECL(2, const libxsmm_bfloat16, bias, ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) ? (libxsmm_bfloat16*) bias_ptr : NULL, cfg.bk);
  LIBXSMM_VLA_DECL(4, __mmask32,  relubitmask, ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) ? (__mmask32*)relu_ptr : NULL, nBlocksOFm, cfg.bn, cfg.bk/32);
  libxsmm_meltwfunction_unary eltwise_kernel_act = cfg.fwd_cvtfp32bf16_relu_kernel;
  libxsmm_meltw_unary_param   eltwise_params_act;
  libxsmm_meltwfunction_unary     eltwise_kernel = cfg.fwd_cvtfp32bf16_kernel;
  libxsmm_meltw_unary_param       eltwise_params;
  libxsmm_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise;
  libxsmm_meltw_unary_param              copy_params;

  unsigned long long  blocks = nBlocksIFm;
  libxsmm_blasint CB_BLOCKS = nBlocksIFm, BF = 1;

  if (((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) && ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU )) {
    bf16_batchreduce_kernel_zerobeta_fused_eltwise = cfg.gemm_fwd7;
  } else if ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) {
    bf16_batchreduce_kernel_zerobeta_fused_eltwise = cfg.gemm_fwd4;
  } else if ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) {
    bf16_batchreduce_kernel_zerobeta_fused_eltwise = cfg.gemm_fwd5;
  }

  BF = cfg.fwd_bf;
  CB_BLOCKS = nBlocksIFm/BF;
  blocks = CB_BLOCKS;

  if (use_2d_blocking == 1) {
    row_teams = cfg.fwd_row_teams;
    column_teams = cfg.fwd_column_teams;
    my_col_id = ltid % column_teams;
    my_row_id = ltid / column_teams;
    im_tasks_per_thread = (nBlocksMB + row_teams-1)/row_teams;
    in_tasks_per_thread = (nBlocksOFm + column_teams-1)/column_teams;
    my_im_start = LIBXSMM_MIN( my_row_id * im_tasks_per_thread, nBlocksMB);
    my_im_end = LIBXSMM_MIN( (my_row_id+1) * im_tasks_per_thread, nBlocksMB);
    my_in_start = LIBXSMM_MIN( my_col_id * in_tasks_per_thread, nBlocksOFm);
    my_in_end = LIBXSMM_MIN( (my_col_id+1) * in_tasks_per_thread, nBlocksOFm);
  }

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  cfg.fwd_config_kernel(NULL, NULL, NULL);

  if (use_2d_blocking == 1) {
    if (BF > 1) {
      for ( ifm1 = 0; ifm1 < BF; ++ifm1 ) {
        for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
          for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
            if ( ifm1 == 0 ) {
              if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
                copy_params.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk);
                copy_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm,cfg.bn,cfg.bk);
                cfg.fwd_colbcast_bf16fp32_copy_kernel(&copy_params);
              } else {
                copy_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                cfg.fwd_zero_kernel(&copy_params);
              }
            }

            cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
                &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);

            if ( ifm1 == BF-1  ) {
              if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
                eltwise_params_act.in.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                eltwise_params_act.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                eltwise_params_act.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
                eltwise_kernel_act(&eltwise_params_act);
              } else {
                eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                eltwise_kernel(&eltwise_params);
              }
            }
          }
        }
      }
    } else {
      if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
        copy_params.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(2, bias, 0, 0,cfg.bk);
        copy_params.out.primary = fp32_bias_scratch;
        cfg.fwd_copy_bf16fp32_kernel(&copy_params);
      }
      for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
        for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
          if ( ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) || ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU )) {
            if ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) {
              gemm_eltwise_params.bias_ptr  = (float*) fp32_bias_scratch + ofm1 * cfg.bk;
            }
            if ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) {
              gemm_eltwise_params.out_ptr   = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
            }
            bf16_batchreduce_kernel_zerobeta_fused_eltwise( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks, &gemm_eltwise_params);
          } else {
            cfg.gemm_fwd3( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks);
          }
        }
      }
    }
  } else {
    if (BF > 1) {
      for ( ifm1 = 0; ifm1 < BF; ++ifm1 ) {
        for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
          mb1  = mb1ofm1%nBlocksMB;
          ofm1 = mb1ofm1/nBlocksMB;
          /* Initialize libxsmm_blasintermediate f32 tensor */
          if ( ifm1 == 0 ) {
            if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
              copy_params.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk);
              copy_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm,cfg.bn,cfg.bk);
              cfg.fwd_colbcast_bf16fp32_copy_kernel(&copy_params);
            } else {
              copy_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              cfg.fwd_zero_kernel(&copy_params);
            }
          }
          cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);

          if ( ifm1 == BF-1  ) {
            if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
              eltwise_params_act.in.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              eltwise_params_act.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              eltwise_params_act.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
              eltwise_kernel_act(&eltwise_params_act);
            } else {
              eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              eltwise_kernel(&eltwise_params);
            }
          }
        }
      }
    } else {
      if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
        copy_params.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(2, bias, 0, 0,cfg.bk);
        copy_params.out.primary = fp32_bias_scratch;
        cfg.fwd_copy_bf16fp32_kernel(&copy_params);
      }
      for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
        mb1  = mb1ofm1%nBlocksMB;
        ofm1 = mb1ofm1/nBlocksMB;
        if ( ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) || ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU )) {
          if ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) {
            gemm_eltwise_params.bias_ptr  = (float*) fp32_bias_scratch + ofm1 * cfg.bk;
          }
          if ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) {
            gemm_eltwise_params.out_ptr   = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
          }
          bf16_batchreduce_kernel_zerobeta_fused_eltwise( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
            &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks, &gemm_eltwise_params);
        } else {
          cfg.gemm_fwd3( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
            &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks);
        }
      }
    }
  }

  cfg.tilerelease_kernel(NULL, NULL, NULL);
  libxsmm_barrier_wait(cfg.barrier, ltid);
}

void my_fc_bwd_exec( my_fc_bwd_config cfg, const libxsmm_bfloat16* wt_ptr, libxsmm_bfloat16* din_act_ptr,
                     const libxsmm_bfloat16* dout_act_ptr, libxsmm_bfloat16* dwt_ptr, const libxsmm_bfloat16* in_act_ptr,
                     libxsmm_bfloat16* dbias_ptr, const unsigned char* relu_ptr, my_pass pass, int start_tid, int my_tid, void* scratch ) {
  /* size variables, all const */
  /* here we assume that input and output blocking is similar */
  const libxsmm_blasint bn = cfg.bn;
  const libxsmm_blasint bk = cfg.bk;
  const libxsmm_blasint bc = cfg.bc;
  libxsmm_blasint lpb = 2;
  const libxsmm_blasint bc_lp = bc/lpb;
  const libxsmm_blasint bk_lp = bk/lpb;
  const libxsmm_blasint bn_lp = bn/lpb;
  const libxsmm_blasint nBlocksIFm = cfg.C / cfg.bc;
  const libxsmm_blasint nBlocksOFm = cfg.K / cfg.bk;
  const libxsmm_blasint nBlocksMB  = cfg.N / cfg.bn;
  libxsmm_blasint mb1ofm1 = 0, mb1 = 0, ofm1 = 0, mb2 = 0, ofm2 = 0;
  libxsmm_blasint iteri = 0, iterj = 0;
  libxsmm_blasint performed_doutput_transpose = 0;
  libxsmm_meltw_transform_param trans_param;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks for transpose that could be run in parallel */
  const libxsmm_blasint eltwise_work = nBlocksOFm * nBlocksMB;
  /* compute chunk size */
  const libxsmm_blasint eltwise_chunksize = (eltwise_work % cfg.threads == 0) ? (eltwise_work / cfg.threads) : ((eltwise_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint eltwise_thr_begin = (ltid * eltwise_chunksize < eltwise_work) ? (ltid * eltwise_chunksize) : eltwise_work;
  const libxsmm_blasint eltwise_thr_end = ((ltid + 1) * eltwise_chunksize < eltwise_work) ? ((ltid + 1) * eltwise_chunksize) : eltwise_work;

  /* number of tasks for transpose that could be run in parallel */
  const libxsmm_blasint dbias_work = nBlocksOFm;
  /* compute chunk size */
  const libxsmm_blasint dbias_chunksize = (dbias_work % cfg.threads == 0) ? (dbias_work / cfg.threads) : ((dbias_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint dbias_thr_begin = (ltid * dbias_chunksize < dbias_work) ? (ltid * dbias_chunksize) : dbias_work;
  const libxsmm_blasint dbias_thr_end = ((ltid + 1) * dbias_chunksize < dbias_work) ? ((ltid + 1) * dbias_chunksize) : dbias_work;

  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, dbias, ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) ? (libxsmm_bfloat16*) dbias_ptr : NULL, cfg.bk);
  LIBXSMM_VLA_DECL(4,     __mmask32, relubitmask, ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) ? (__mmask32*)relu_ptr : NULL, nBlocksOFm, cfg.bn, cfg.bk/32);

  libxsmm_bfloat16 *grad_output_ptr = (((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU)) ? (libxsmm_bfloat16*)((char*)scratch + cfg.doutput_scratch_mark) : (libxsmm_bfloat16*)dout_act_ptr;
  libxsmm_bfloat16 *tr_doutput_ptr = (((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU)) ? (libxsmm_bfloat16*)grad_output_ptr + cfg.N * cfg.K : (libxsmm_bfloat16*)scratch;
  LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16,   doutput_orig, (libxsmm_bfloat16*)dout_act_ptr, nBlocksOFm, bn, bk);
  libxsmm_meltw_unary_param   relu_params;
  libxsmm_meltwfunction_unary relu_kernel = cfg.bwd_relu_kernel;
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16,   doutput, grad_output_ptr, nBlocksOFm, bn, bk);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, doutput_tr, tr_doutput_ptr, nBlocksMB, bn_lp, bk, lpb);

  libxsmm_meltwfunction_unary eltwise_kernel  = cfg.bwd_cvtfp32bf16_kernel;
  libxsmm_meltwfunction_unary eltwise_kernel2 = cfg.upd_cvtfp32bf16_kernel;
  libxsmm_meltw_unary_param   eltwise_params;
  libxsmm_meltw_unary_param          copy_params;
  libxsmm_meltw_unary_param        delbias_params;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);
  cfg.bwd_config_kernel(NULL, NULL, NULL);

  /* Apply to doutput potential fusions */
  if (((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU)) {
    for ( mb1ofm1 = eltwise_thr_begin; mb1ofm1 < eltwise_thr_end; ++mb1ofm1 ) {
      mb1  = mb1ofm1/nBlocksOFm;
      ofm1 = mb1ofm1%nBlocksOFm;
      relu_params.in.primary   = (void*) &LIBXSMM_VLA_ACCESS(4, doutput_orig, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      relu_params.out.primary  = &LIBXSMM_VLA_ACCESS(4, doutput, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      relu_params.in.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
      relu_kernel(&relu_params);

      /* If in UPD pass, also perform transpose of doutput  */
      if ( (pass & MY_PASS_BWD_W) == MY_PASS_BWD_W ) {
        trans_param.in_ptr  = &LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, 0, 0, nBlocksOFm, bn, bk);
        trans_param.out_ptr = &LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, 0, 0, 0, nBlocksMB, bn_lp, bk, lpb);
        cfg.norm_to_vnni_kernel(&trans_param);
      }
    }

    if ( (pass & MY_PASS_BWD_W) == MY_PASS_BWD_W ) {
      performed_doutput_transpose = 1;
    }
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }


  /* Accumulation of bias happens in f32 */
  if (((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS)) {
    for ( ofm1 = dbias_thr_begin; ofm1 < dbias_thr_end; ++ofm1 ) {
      delbias_params.in.primary     = &LIBXSMM_VLA_ACCESS(4,  doutput, 0, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      delbias_params.out.primary = &LIBXSMM_VLA_ACCESS(2,  dbias, ofm1, 0, cfg.bk);
      cfg.delbias_reduce_kernel(&delbias_params);
    }
    /* wait for eltwise to finish */
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if ( (pass & MY_PASS_BWD_D) == MY_PASS_BWD_D ){
    libxsmm_blasint use_2d_blocking = cfg.bwd_2d_blocking;

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
    libxsmm_blasint ifm1 = 0, ifm2 = 0, ifm1ofm1 = 0, mb1ifm1 = 0;
    libxsmm_blasint im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;

    LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, filter, (libxsmm_bfloat16*)wt_ptr, nBlocksIFm, bc_lp, bk, lpb);
    LIBXSMM_VLA_DECL(4,        libxsmm_bfloat16,    dinput, (libxsmm_bfloat16* )din_act_ptr, nBlocksIFm, bn, bc);
    LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, filter_tr, (libxsmm_bfloat16*)scratch, nBlocksOFm, bk_lp, bc, lpb);
    float* temp_output = (float*)scratch + (cfg.C * cfg.K)/2;
    LIBXSMM_VLA_DECL(4,        float,    dinput_f32, (float*) temp_output, nBlocksIFm, bn, bc);

    unsigned long long  blocks = nBlocksOFm;
    libxsmm_blasint KB_BLOCKS = nBlocksOFm, BF = 1;
    BF = cfg.bwd_bf;
    KB_BLOCKS = nBlocksOFm/BF;
    blocks = KB_BLOCKS;

    if (use_2d_blocking == 1) {
      row_teams = cfg.bwd_row_teams;
      column_teams = cfg.bwd_column_teams;
      my_col_id = ltid % column_teams;
      my_row_id = ltid / column_teams;
      im_tasks_per_thread = (nBlocksMB + row_teams-1)/row_teams;
      in_tasks_per_thread = (nBlocksIFm + column_teams-1)/column_teams;
      my_im_start = LIBXSMM_MIN( my_row_id * im_tasks_per_thread, nBlocksMB);
      my_im_end = LIBXSMM_MIN( (my_row_id+1) * im_tasks_per_thread, nBlocksMB);
      my_in_start = LIBXSMM_MIN( my_col_id * in_tasks_per_thread, nBlocksIFm);
      my_in_end = LIBXSMM_MIN( (my_col_id+1) * in_tasks_per_thread, nBlocksIFm);
    }

    /* transpose weight */
    for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
      ofm1 = ifm1ofm1 / nBlocksIFm;
      ifm1 = ifm1ofm1 % nBlocksIFm;
      trans_param.in_ptr  = &LIBXSMM_VLA_ACCESS(5, filter,  ofm1, ifm1, 0, 0, 0, nBlocksIFm, bc_lp, bk, lpb);
      trans_param.out_ptr = &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb);
      cfg.vnni_to_vnniT_kernel(&trans_param);
    }

    /* wait for transpose to finish */
    libxsmm_barrier_wait(cfg.barrier, ltid);

    if (use_2d_blocking == 1) {
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for (ifm1 = my_in_start; ifm1 < my_in_end; ++ifm1) {
            for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
              /* Initialize libxsmm_blasintermediate f32 tensor */
              if ( ofm1 == 0 ) {
                copy_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dinput_f32, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
                cfg.bwd_zero_kernel(&copy_params);
              }
              cfg.gemm_bwd( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb),
                  &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
                  &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
              /* downconvert libxsmm_blasintermediate f32 tensor to bf 16 and store to final C */
              if ( ofm1 == BF-1  ) {
                eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
                eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
                eltwise_kernel(&eltwise_params);
              }
            }
          }
        }
      } else {
        for (ifm1 = my_in_start; ifm1 < my_in_end; ++ifm1) {
          for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
            cfg.gemm_bwd3( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, 0, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb),
                &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
          }
        }
      }
    } else {
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
            mb1  = mb1ifm1%nBlocksMB;
            ifm1 = mb1ifm1/nBlocksMB;
            /* Initialize libxsmm_blasintermediate f32 tensor */
            if ( ofm1 == 0 ) {
              copy_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dinput_f32, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
              cfg.bwd_zero_kernel(&copy_params);
            }
            cfg.gemm_bwd( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb),
                &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
            /* downconvert libxsmm_blasintermediate f32 tensor to bf 16 and store to final C */
            if ( ofm1 == BF-1  ) {
                eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
                eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
                eltwise_kernel(&eltwise_params);
            }
          }
        }
      } else {
        for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
          mb1  = mb1ifm1%nBlocksMB;
          ifm1 = mb1ifm1/nBlocksMB;
          cfg.gemm_bwd3( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, 0, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb),
              &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk),
              &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
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
    libxsmm_blasint im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;

    /* compute chunk size */
    const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
    libxsmm_blasint BF = cfg.upd_bf;

    /* loop variables */
    libxsmm_blasint ifm1ofm1 = 0, ifm1 = 0, ifm2 = 0, bfn = 0, ii = 0, jj = 0, mb1ifm1 = 0, jc = 0, jk = 0;

    /* Batch reduce related variables */
    unsigned long long  blocks = nBlocksMB/BF;

    LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16,  input,    (libxsmm_bfloat16* )in_act_ptr, nBlocksIFm, bn, bc);
    LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, dfilter,  (libxsmm_bfloat16*)dwt_ptr, nBlocksIFm, bc_lp, bk, lpb);

    /* Set up tensors for transposing/scratch before vnni reformatting dfilter */
    libxsmm_bfloat16  *tr_inp_ptr = (libxsmm_bfloat16*) ((libxsmm_bfloat16*)scratch + cfg.N * cfg.K);
    float               *dfilter_f32_ptr = (float*) ((libxsmm_bfloat16*)tr_inp_ptr + cfg.N * cfg.C);
    libxsmm_bfloat16 *dfilter_scratch = (libxsmm_bfloat16*) ((float*)dfilter_f32_ptr + cfg.C * cfg.K) + ltid * bc * bk;

    LIBXSMM_VLA_DECL(4, libxsmm_bfloat16,  input_tr,    (libxsmm_bfloat16*)tr_inp_ptr, nBlocksMB, bc, bn);
    LIBXSMM_VLA_DECL(4,       float, dfilter_f32,  (float*)dfilter_f32_ptr, nBlocksIFm, bc, bk);
    LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, dfilter_block,  (libxsmm_bfloat16*)dfilter_scratch, bk);

    const libxsmm_blasint tr_out_work = nBlocksMB * nBlocksOFm;
    const libxsmm_blasint tr_out_chunksize = (tr_out_work % cfg.threads == 0) ? (tr_out_work / cfg.threads) : ((tr_out_work / cfg.threads) + 1);
    const libxsmm_blasint tr_out_thr_begin = (ltid * tr_out_chunksize < tr_out_work) ? (ltid * tr_out_chunksize) : tr_out_work;
    const libxsmm_blasint tr_out_thr_end = ((ltid + 1) * tr_out_chunksize < tr_out_work) ? ((ltid + 1) * tr_out_chunksize) : tr_out_work;

    const libxsmm_blasint tr_inp_work = nBlocksMB * nBlocksIFm;
    const libxsmm_blasint tr_inp_chunksize = (tr_inp_work % cfg.threads == 0) ? (tr_inp_work / cfg.threads) : ((tr_inp_work / cfg.threads) + 1);
    const libxsmm_blasint tr_inp_thr_begin = (ltid * tr_inp_chunksize < tr_inp_work) ? (ltid * tr_inp_chunksize) : tr_inp_work;
    const libxsmm_blasint tr_inp_thr_end = ((ltid + 1) * tr_inp_chunksize < tr_inp_work) ? ((ltid + 1) * tr_inp_chunksize) : tr_inp_work;

    /* These are used for the vnni reformatting of the f32 output  */
    __m512 a01, b01;
    __m512i c01 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32();
    const __m512i perm_index = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);

    if (use_2d_blocking == 1) {
      row_teams = cfg.upd_row_teams;
      column_teams = cfg.upd_column_teams;
      my_col_id = ltid % column_teams;
      my_row_id = ltid / column_teams;
      im_tasks_per_thread = (nBlocksIFm + row_teams-1)/row_teams;
      in_tasks_per_thread = (nBlocksOFm + column_teams-1)/column_teams;
      my_im_start = LIBXSMM_MIN( my_row_id * im_tasks_per_thread, nBlocksIFm);
      my_im_end = LIBXSMM_MIN( (my_row_id+1) * im_tasks_per_thread, nBlocksIFm);
      my_in_start = LIBXSMM_MIN( my_col_id * in_tasks_per_thread, nBlocksOFm);
      my_in_end = LIBXSMM_MIN( (my_col_id+1) * in_tasks_per_thread, nBlocksOFm);
    }

    /* Required upfront tranposes */
    for (mb1ifm1 = tr_inp_thr_begin; mb1ifm1 < tr_inp_thr_end; mb1ifm1++) {
      mb1 = mb1ifm1%nBlocksMB;
      ifm1 = mb1ifm1/nBlocksMB;
      trans_param.in_ptr  = &LIBXSMM_VLA_ACCESS(4, input, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
      trans_param.out_ptr = &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, mb1, 0, 0, nBlocksMB, bc, bn);
      cfg.norm_to_normT_kernel(&trans_param);
    }

    if (performed_doutput_transpose == 0) {
      for (mb1ofm1 = tr_out_thr_begin; mb1ofm1 < tr_out_thr_end; mb1ofm1++) {
        mb1 = mb1ofm1%nBlocksMB;
        ofm1 = mb1ofm1/nBlocksMB;
        trans_param.in_ptr  = &LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, 0, 0, nBlocksOFm, bn, bk);
        trans_param.out_ptr = &LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, 0, 0, 0, nBlocksMB, bn_lp, bk, lpb);
        cfg.norm_to_vnni_kernel(&trans_param);
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    if (use_2d_blocking == 1) {
      ifm2 = 0;
      ofm2 = 0;
      if (BF == 1) {
        for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
          for (ifm1 = my_im_start; ifm1 < my_im_end; ++ifm1) {
            cfg.gemm_upd3(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, 0, 0, ofm2*bbk, 0, nBlocksMB, bn_lp, bk, lpb), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, 0, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, 0, 0, 0, nBlocksIFm, bc_lp, bk, lpb), &blocks);
          }
        }
      } else {
        for (bfn = 0; bfn < BF; bfn++) {
          for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
            for (ifm1 = my_im_start; ifm1 < my_im_end; ++ifm1) {
              /* initialize current work task to zero */
              if (bfn == 0) {
                copy_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk);
                cfg.upd_zero_kernel(&copy_params);
              }
              cfg.gemm_upd(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, bfn*blocks, 0, ofm2*bbk, 0, nBlocksMB, bn_lp, bk, lpb), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, bfn*blocks, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
              /* Downconvert result to BF16 and vnni format */
              if (bfn == BF-1) {
                LIBXSMM_ALIGNED(libxsmm_bfloat16 tmp_buf[bc][bk], 64);
                eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
                eltwise_params.out.primary = tmp_buf;
                trans_param.in_ptr  = tmp_buf;
                trans_param.out_ptr = &LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, 0, 0, 0, nBlocksIFm, bc_lp, bk, lpb);
                eltwise_kernel2(&eltwise_params);
                cfg.norm_to_vnni_kernel_wt(&trans_param);
              }
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
          cfg.gemm_upd3(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, 0, 0, ofm2*bbk, 0, nBlocksMB, bn_lp, bk, lpb), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, 0, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc)/lpb, ofm2*bbk, 0, nBlocksIFm, bc_lp, bk, lpb), &blocks);
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
              copy_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk);
              cfg.upd_zero_kernel(&copy_params);
            }
            cfg.gemm_upd(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, bfn*blocks, 0, ofm2*bbk, 0, nBlocksMB, bn_lp, bk, lpb), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, bfn*blocks, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
            /* Downconvert result to BF16 and vnni format */
            if (bfn == BF-1) {
              LIBXSMM_ALIGNED(libxsmm_bfloat16 tmp_buf[bc][bk], 64);
              eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk);
              eltwise_params.out.primary = tmp_buf;
              trans_param.in_ptr  = tmp_buf;
              trans_param.out_ptr = &LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc)/lpb, ofm2*bbk, 0, nBlocksIFm, bc_lp, bk, lpb);
              eltwise_kernel2(&eltwise_params);
              cfg.norm_to_vnni_kernel_wt(&trans_param);
            }
          }
        }
      }
    }
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }
  cfg.tilerelease_kernel(NULL, NULL, NULL);
}

int main(int argc, char* argv[])
{
  float *naive_input, *naive_output, *naive_filter, *naive_delinput, *naive_deloutput, *naive_delfilter, *naive_bias, *naive_delbias;
  libxsmm_bfloat16 *naive_input_bf16, *naive_filter_bf16, *naive_output_bf16, *naive_delinput_bf16, *naive_delfilter_bf16, *naive_deloutput_bf16, *naive_bias_bf16, *naive_delbias_bf16;
  float *naive_libxsmm_output_f32, *naive_libxsmm_delinput_f32, *naive_libxsmm_delfilter_f32, *naive_libxsmm_delbias_f32;
  libxsmm_bfloat16 *naive_libxsmm_output_bf16, *naive_libxsmm_delinput_bf16, *naive_libxsmm_delfilter_bf16, *naive_libxsmm_delbias_bf16;
  libxsmm_bfloat16 *input_libxsmm, *filter_libxsmm, *delinput_libxsmm, *delfilter_libxsmm, *output_libxsmm, *deloutput_libxsmm, *bias_libxsmm, *delbias_libxsmm;
  unsigned char *relumask_libxsmm;
  my_eltwise_fuse my_fuse;
  my_fc_fwd_config my_fc_fwd;
  my_fc_bwd_config my_fc_bwd;
  naive_fullyconnected_t naive_param;
  void* scratch = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int nImg = 256;          /* mini-batch size, "N" */
  int nIFm = 1024;          /* number of input feature maps, "C" */
  int nOFm = 1024;          /* number of input feature maps, "C" */
  int fuse_type = 0;      /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  int bn = 32;
  int bk = 32;
  int bc = 32;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double gflop = 0.0;
  int i;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&norms_upd);
  libxsmm_matdiff_clear(&diff);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters nImg nIFm nOFm fuse_type type bn bk bc\n", argv[0]);
    return 0;
  }
  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) nImg       = atoi(argv[i++]);
  if (argc > i) nIFm       = atoi(argv[i++]);
  if (argc > i) nOFm       = atoi(argv[i++]);
  if (argc > i) fuse_type  = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);
  if (argc > i) bn         = atoi(argv[i++]);
  if (argc > i) bk         = atoi(argv[i++]);
  if (argc > i) bc         = atoi(argv[i++]);

  if (type != 'A' && type != 'F' && type != 'B' && type != 'U' && type != 'M') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only), 'U' (UP only). 'M' (BPUP-fused only)\n");
    return -1;
  }
  if ( (fuse_type < 0) || (fuse_type > 4) || (fuse_type == 3) ) {
    printf("fuse type needs to be 0 (None), 1 (Bias), 2 (ReLU), 4 (Bias+ReLU)\n");
    return -1;
  }

  /* set struct for naive convolution */
  naive_param.N = nImg;
  naive_param.C = nIFm;
  naive_param.K = nOFm;
  naive_param.fuse_type = fuse_type;

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: N:%d  C:%d  K:%d\n", nImg, nIFm, nOFm);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIFm*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOFm*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIFm*   sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOFm*   sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Filter     : %10.2f MiB\n", (double)(nIFm*nOFm*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );

  /* allocate data */
  naive_input                 = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_delinput              = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_output                = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_deloutput             = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_filter                = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_delfilter             = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_bias                  = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);
  naive_delbias               = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);

  naive_input_bf16            = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_delinput_bf16         = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_output_bf16           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_deloutput_bf16        = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_filter_bf16           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_delfilter_bf16        = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_bias_bf16             = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);
  naive_delbias_bf16          = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);

  naive_libxsmm_output_bf16   = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_delinput_bf16 = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_delfilter_bf16= (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_delbias_bf16  = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_output_f32    = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_libxsmm_delinput_f32  = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_libxsmm_delfilter_f32 = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_libxsmm_delbias_f32   = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);

  input_libxsmm               = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  delinput_libxsmm            = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  output_libxsmm              = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  deloutput_libxsmm           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  filter_libxsmm              = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  delfilter_libxsmm           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  bias_libxsmm                =  (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);
  delbias_libxsmm             =  (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);
  relumask_libxsmm            =  (unsigned char*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(unsigned char), 2097152);

  /* initialize data */
  init_buf( naive_input,     nImg*nIFm, 0, 0 );
  init_buf( naive_delinput,  nImg*nIFm, 0, 0 );
  init_buf( naive_output,    nImg*nOFm, 0, 0 );
  init_buf( naive_deloutput, nImg*nOFm, 0, 0 );
  init_buf( naive_filter,    nIFm*nOFm, 0, 0 );
  init_buf( naive_delfilter, nIFm*nOFm, 0, 0 );
  init_buf( naive_bias,      nOFm,      0, 0 );
  init_buf( naive_delbias,   nOFm,      0, 0 );

  libxsmm_rne_convert_fp32_bf16( naive_input,     naive_input_bf16,     nImg*nIFm );
  libxsmm_rne_convert_fp32_bf16( naive_delinput,  naive_delinput_bf16,  nImg*nIFm );
  libxsmm_rne_convert_fp32_bf16( naive_output,    naive_output_bf16,    nImg*nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_deloutput, naive_deloutput_bf16, nImg*nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_filter,    naive_filter_bf16,    nIFm*nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_delfilter, naive_delfilter_bf16, nIFm*nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_bias,      naive_bias_bf16,      nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_delbias,   naive_delbias_bf16,   nOFm );

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    if (type == 'A' || type == 'F') {
      naive_fullyconnected_fused_fp(&naive_param, naive_input, naive_output, naive_filter, naive_bias);
    }
    if (type == 'A' || type == 'B' || type == 'M') {
      naive_fullyconnected_fused_bp(&naive_param, naive_delinput, naive_deloutput, naive_filter, naive_delbias, naive_output);
    }
    if (type == 'A' || type == 'U' || type == 'M') {
      naive_fullyconnected_wu(&naive_param, naive_input, naive_deloutput, naive_delfilter);
    }
    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  printf("\n");
  printf("##########################################\n");
  printf("#      Setting Up  (custom-Storage)      #\n");
  printf("##########################################\n");

  if (nImg % bn != 0) {
    bn = nImg;
  }
  if (nIFm % bc != 0) {
    bc = nIFm;
  }
  if (nOFm % bk != 0) {
    bk = nOFm;
  }

  if ( fuse_type == 0 ) {
    my_fuse = MY_ELTWISE_FUSE_NONE;
  } else if ( fuse_type == 1 ) {
    my_fuse = MY_ELTWISE_FUSE_BIAS;
  } else if ( fuse_type == 2 ) {
    my_fuse = MY_ELTWISE_FUSE_RELU;
  } else if ( fuse_type == 4 ) {
    my_fuse = MY_ELTWISE_FUSE_BIAS_RELU;
  } else {
    /* cannot happen */
  }

  my_fc_fwd = setup_my_fc_fwd(nImg, nIFm, nOFm, bn, bc, bk, nThreads, my_fuse);
  my_fc_bwd = setup_my_fc_bwd(nImg, nIFm, nOFm, bn, bc, bk, nThreads, my_fuse);

  /* we can also use the layout functions and set the data on our
     own external to the library */
   matrix_copy_NC_to_NCNC_bf16( naive_input_bf16,     input_libxsmm,     1, nImg, nIFm, bn, bc );
  matrix_copy_NC_to_NCNC_bf16( naive_delinput_bf16,  delinput_libxsmm,  1, nImg, nIFm, bn, bc );
  matrix_copy_NC_to_NCNC_bf16( naive_output_bf16,    output_libxsmm,    1, nImg, nOFm, bn, bk );
  matrix_copy_NC_to_NCNC_bf16( naive_deloutput_bf16, deloutput_libxsmm, 1, nImg, nOFm, bn, bk );
  matrix_copy_KC_to_KCCK_bf16( naive_filter_bf16,    filter_libxsmm      , nIFm, nOFm, bc, bk );
  matrix_copy_KC_to_KCCK_bf16( naive_delfilter_bf16, delfilter_libxsmm   , nIFm, nOFm, bc, bk );
  matrix_copy_NC_to_NCNC_bf16( naive_bias_bf16,    bias_libxsmm,    1, 1, nOFm, 1, nOFm );
  matrix_copy_NC_to_NCNC_bf16( naive_delbias_bf16, delbias_libxsmm, 1, 1, nOFm, 1, nOFm );


  /* let's allocate and bind scratch */
  if ( my_fc_fwd.scratch_size > 0 || my_fc_bwd.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( my_fc_fwd.scratch_size, my_fc_bwd.scratch_size);
    scratch = libxsmm_aligned_scratch( alloc_size, 2097152 );
    init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
  }

  if ((type == 'A' || type == 'F') && LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#   Correctness - FWD (custom-Storage)   #\n");
    printf("##########################################\n");

#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_fc_fwd_exec( my_fc_fwd, filter_libxsmm, input_libxsmm, output_libxsmm,
                      bias_libxsmm, relumask_libxsmm, 0, tid, scratch );
    }

    /* copy out data */
    matrix_copy_NCNC_to_NC_bf16( output_libxsmm, naive_libxsmm_output_bf16, 1, nImg, nOFm, bn, bk );
    libxsmm_convert_bf16_f32( naive_libxsmm_output_bf16, naive_libxsmm_output_f32, nImg*nOFm );

    /* compare */
    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nOFm, 1, naive_output, naive_libxsmm_output_f32, 0, 0);
    printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_fwd);
  }

  if ( (type == 'A' || type == 'M') && LIBXSMM_NEQ(0, check) ) {
    printf("##########################################\n");
    printf("# Correctness - BWDUPD (custom-Storage)  #\n");
    printf("##########################################\n");

#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_fc_bwd_exec( my_fc_bwd, filter_libxsmm, delinput_libxsmm, deloutput_libxsmm, delfilter_libxsmm,
                      input_libxsmm, delbias_libxsmm, relumask_libxsmm, MY_PASS_BWD, 0, tid, scratch );
    }

    /* copy out data */
    matrix_copy_NCNC_to_NC_bf16( delinput_libxsmm, naive_libxsmm_delinput_bf16, 1, nImg, nIFm, bn, bc );
    libxsmm_convert_bf16_f32( naive_libxsmm_delinput_bf16, naive_libxsmm_delinput_f32, nImg*nIFm );
    /* copy out data */
    matrix_copy_KCCK_to_KC_bf16( delfilter_libxsmm, naive_libxsmm_delfilter_bf16, nIFm, nOFm, bc, bk );
    libxsmm_convert_bf16_f32( naive_libxsmm_delfilter_bf16, naive_libxsmm_delfilter_f32, nIFm*nOFm );

    /* compare */
    libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nImg*nIFm, 1, naive_delinput, naive_libxsmm_delinput_f32, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_bwd);
    if ( (fuse_type == 1) || (fuse_type == 4) ) {
      matrix_copy_NCNC_to_NC_bf16( delbias_libxsmm, naive_libxsmm_delbias_bf16, 1, 1, nOFm, 1, nOFm );
      libxsmm_convert_bf16_f32( naive_libxsmm_delbias_bf16, naive_libxsmm_delbias_f32, nOFm );
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nOFm, 1, naive_delbias, naive_libxsmm_delbias_f32, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
    }

    libxsmm_matdiff(&norms_upd, LIBXSMM_DATATYPE_F32, nIFm*nOFm, 1, naive_delfilter, naive_libxsmm_delfilter_f32, 0, 0);
    printf("L1 reference  : %.25g\n", norms_upd.l1_ref);
    printf("L1 test       : %.25g\n", norms_upd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_upd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_upd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_upd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_upd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_upd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_upd);
  }

  if (type == 'A' || type == 'F') {
    printf("##########################################\n");
    printf("#   Performance - FWD (custom-Storage)   #\n");
    printf("##########################################\n");
    l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      for (i = 0; i < iters; ++i) {
        my_fc_fwd_exec( my_fc_fwd, filter_libxsmm, input_libxsmm, output_libxsmm,
                        bias_libxsmm, relumask_libxsmm, 0, tid, scratch );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = (2.0*(double)nImg*(double)nIFm*(double)nOFm*(double)iters) / (1000*1000*1000);

    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);

    printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIFm,
        nOFm, ((double)(l_total/iters)), gflop/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
        norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
  }

  if (type == 'A' || type == 'M') {
    printf("##########################################\n");
    printf("# Performance - BWDUPD (custom-Storage)  #\n");
    printf("##########################################\n");
    l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      for (i = 0; i < iters; ++i) {
        my_fc_bwd_exec( my_fc_bwd, filter_libxsmm, delinput_libxsmm, deloutput_libxsmm, delfilter_libxsmm,
                        input_libxsmm, delbias_libxsmm, relumask_libxsmm, MY_PASS_BWD, 0, tid, scratch );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = (4.0*(double)nImg*(double)nIFm*(double)nOFm*(double)iters) / (1000*1000*1000);

    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);

    printf("PERFDUMP,UP,%s,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIFm,
        nOFm, ((double)(l_total/iters)), gflop/l_total, norms_upd.l1_ref, norms_upd.l1_tst,
        norms_upd.l2_abs, norms_upd.l2_rel, norms_upd.linf_abs, norms_upd.linf_rel, norms_upd.normf_rel);
  }

  /* deallocate data */
  if ( scratch != NULL ) {
    libxsmm_free(scratch);
  }
  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_delinput);
  libxsmm_free(naive_deloutput);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_delfilter);
  libxsmm_free(naive_input_bf16);
  libxsmm_free(naive_delinput_bf16);
  libxsmm_free(naive_output_bf16);
  libxsmm_free(naive_deloutput_bf16);
  libxsmm_free(naive_filter_bf16);
  libxsmm_free(naive_delfilter_bf16);
  libxsmm_free(naive_libxsmm_output_bf16);
  libxsmm_free(naive_libxsmm_delinput_bf16);
  libxsmm_free(naive_libxsmm_delfilter_bf16);
  libxsmm_free(naive_libxsmm_output_f32);
  libxsmm_free(naive_libxsmm_delinput_f32);
  libxsmm_free(naive_libxsmm_delfilter_f32);
  libxsmm_free(input_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(delinput_libxsmm);
  libxsmm_free(deloutput_libxsmm);
  libxsmm_free(filter_libxsmm);
  libxsmm_free(delfilter_libxsmm);
  libxsmm_free(naive_bias);
  libxsmm_free(naive_delbias);
  libxsmm_free(naive_bias_bf16);
  libxsmm_free(naive_delbias_bf16);
  libxsmm_free(naive_libxsmm_delbias_bf16);
  libxsmm_free(naive_libxsmm_delbias_f32);
  libxsmm_free(relumask_libxsmm);
  libxsmm_free(bias_libxsmm);
  libxsmm_free(delbias_libxsmm);

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 1.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return 0;
}

