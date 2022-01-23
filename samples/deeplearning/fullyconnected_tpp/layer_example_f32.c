/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
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
  libxsmm_blasint fwd_col_teams;
  /* TODO: add hyperpartitions support */
  libxsmm_blasint fwd_M_hyperpartitions;
  libxsmm_blasint fwd_N_hyperpartitions;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
  libxsmm_smmfunction_reducebatch_strd gemm_fwd;
  libxsmm_smmfunction_reducebatch_strd gemm_fwd2;
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
  libxsmm_smmfunction_reducebatch_strd gemm_bwd;
  libxsmm_smmfunction_reducebatch_strd gemm_bwd2;
  libxsmm_smmfunction_reducebatch_strd gemm_upd;
  libxsmm_smmfunction_reducebatch_strd gemm_upd2;
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
  res.gemm_fwd  = libxsmm_smmdispatch_reducebatch_strd(res.bk, res.bn, res.bc,
      res.bk*res.bc*sizeof(float), res.bc*res.bn*sizeof(float),
      &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
  if ( res.gemm_fwd == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_fwd2 = libxsmm_smmdispatch_reducebatch_strd(res.bk, res.bn, res.bc,
      res.bk*res.bc*sizeof(float), res.bc*res.bn*sizeof(float),
      &lda, &ldb, &ldc, &alpha, &zerobeta, NULL, NULL);
  if ( res.gemm_fwd2 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd2 failed. Bailing...!\n");
    exit(-1);
  }

  /* Eltwise TPPs  */
  res.fwd_zero_kernel = libxsmm_dispatch_meltw_unary(bn*bk, 1, &ld_zero, &ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.fwd_zero_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.fwd_relu_kernel  = libxsmm_dispatch_meltw_unary(res.bk, res.bn, &ldc, &ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT, LIBXSMM_MELTW_TYPE_UNARY_RELU);
  if ( res.fwd_relu_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_relu_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.fwd_colbcast_copy_kernel = libxsmm_dispatch_meltw_unary(bk, bn, &ldc, &ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY );
  if ( res.fwd_colbcast_copy_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_colbcast_bf16fp32_copy_kernel failed. Bailing...!\n");
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
  /* BWD GEMM */
  res.gemm_bwd  = libxsmm_smmdispatch_reducebatch_strd(res.bc, res.bn, res.bk,
      res.bk*res.bc*sizeof(float), res.bk*res.bn*sizeof(float),
      &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
  if ( res.gemm_bwd == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_bwd2 = libxsmm_smmdispatch_reducebatch_strd(res.bc, res.bn, res.bk,
      res.bk*res.bc*sizeof(float), res.bk*res.bn*sizeof(float),
      &lda, &ldb, &ldc, &alpha, &zerobeta, NULL, NULL);
  if ( res.gemm_bwd2 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd2 failed. Bailing...!\n");
    exit(-1);
  }

  res.norm_to_normT_kernel = libxsmm_dispatch_meltw_unary(bk, bc, &ldaT, &lda, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
  if ( res.norm_to_normT_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP norm_to_normT_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.bwd_relu_kernel   = libxsmm_dispatch_meltw_unary(bk, bn, &ld_relu_bwd, &ld_relu_bwd, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT, LIBXSMM_MELTW_TYPE_UNARY_RELU_INV);
  if ( res.bwd_relu_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP bwd_relu_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.bwd_zero_kernel = libxsmm_dispatch_meltw_unary(bn*bc, 1, &ld_zero_bwd, &ld_zero_bwd, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
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
  res.gemm_upd = libxsmm_smmdispatch_reducebatch_strd(updM, updN, res.bn,
      res.K*res.bn*sizeof(float), res.C*res.bn*sizeof(float),
      &lda, &ldb, &ldc, &alpha, &beta, &updflags, NULL);
  if ( res.gemm_upd == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_upd failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_upd2 = libxsmm_smmdispatch_reducebatch_strd(updM, updN, res.bn,
      res.K*res.bn*sizeof(float), res.C*res.bn*sizeof(float),
      &lda, &ldb, &ldc, &alpha, &zerobeta, &updflags, NULL);
  if ( res.gemm_upd2 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_upd2 failed. Bailing...!\n");
    exit(-1);
  }

  /* JIT TPP kernels */
  res.upd_zero_kernel = libxsmm_dispatch_meltw_unary(bk, bc, &ld_zero_upd, &ld_zero_upd, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.upd_zero_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP upd_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.delbias_reduce_kernel = libxsmm_dispatch_meltw_unary(bk, bn, &delbias_K, &delbias_N, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT);
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
            /* BRGEMM */
            cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
                &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
            /* apply post BRGEMM fusion */
            if ( ifm1 == BF-1  ) {
              if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
                eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                eltwise_params.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
                cfg.fwd_relu_kernel(&eltwise_params);
              }
            }
          }
        }
      }
    } else {
      for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
        for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
          if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
            eltwise_params.in.primary  = (void*) &LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk);
            eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            cfg.fwd_colbcast_copy_kernel(&eltwise_params);
            cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
                &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
          } else {
            cfg.gemm_fwd2( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
                &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
          }
          /* post GEMM fusion */
          if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
            eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            eltwise_params.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
            cfg.fwd_relu_kernel(&eltwise_params);
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
          /* BRGEMM */
          cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
          /* post GEMM fusion */
          if ( ifm1 == BF-1  ) {
            if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
              eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              eltwise_params.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
              cfg.fwd_relu_kernel(&eltwise_params);
            }
          }
        }
      }
    } else {
      for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
        mb1  = mb1ofm1%nBlocksMB;
        ofm1 = mb1ofm1/nBlocksMB;
        if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
          eltwise_params.in.primary  = (void*) &LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk);
          eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          cfg.fwd_colbcast_copy_kernel(&eltwise_params);
          cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
        } else {
          cfg.gemm_fwd2( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
        }
        /* post GEMM fusion */
        if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
          eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          eltwise_params.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
          cfg.fwd_relu_kernel(&eltwise_params);
        }
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
              cfg.gemm_bwd( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bk, bc ),
                  &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
                  &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
            }
          }
        }
      } else {
        for (ifm1 = my_M_start; ifm1 < my_M_end; ++ifm1) {
          for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
            cfg.gemm_bwd2( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, 0, 0, 0, nBlocksOFm, bk, bc),
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
            /* Initialize intermediate f32 tensor */
            if ( ofm1 == 0 ) {
              eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dinput, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
              cfg.bwd_zero_kernel(&eltwise_params);
            }
            cfg.gemm_bwd( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bk, bc ),
                &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
          }
        }
      } else {
        for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
          mb1  = mb1ifm1%nBlocksMB;
          ifm1 = mb1ifm1/nBlocksMB;
          cfg.gemm_bwd2( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, 0, 0, 0, nBlocksOFm, bk, bc ),
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
            cfg.gemm_upd2(&LIBXSMM_VLA_ACCESS(4, doutput, 0, ofm1, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, input,   0, ifm1, 0, 0, nBlocksIFm, bn, bc),
                &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk), &blocks);
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
              cfg.gemm_upd( &LIBXSMM_VLA_ACCESS(4, doutput, bfn*blocks, ofm1, 0, 0, nBlocksOFm, bn, bk),
                  &LIBXSMM_VLA_ACCESS(4, input,   bfn*blocks, ifm1, 0, 0, nBlocksIFm, bn, bc),
                  &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk), &blocks);
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

          cfg.gemm_upd2( &LIBXSMM_VLA_ACCESS(4, doutput, 0, ofm1, 0, ofm2*bbk, nBlocksOFm, bn, bk),
              &LIBXSMM_VLA_ACCESS(4, input,   0, ifm1, 0, ifm2*bbc, nBlocksIFm, bn, bc),
              &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
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

            cfg.gemm_upd( &LIBXSMM_VLA_ACCESS(4, doutput, bfn*blocks, ofm1, 0, ofm2*bbk, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, input,   bfn*blocks, ifm1, 0, ifm2*bbc, nBlocksIFm, bn, bc),
                &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
          }
        }
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);
  }
}

int main(int argc, char* argv[])
{
  float *naive_input, *naive_output, *naive_filter, *naive_delinput, *naive_deloutput, *naive_delfilter;
  float *naive_bias, *naive_delbias, *naive_deloutput_copy;
  float *naive_libxsmm_output, *naive_libxsmm_delinput, *naive_libxsmm_delfilter;
  float *input_libxsmm, *output_libxsmm, *filter_libxsmm, *delinput_libxsmm, *deloutput_libxsmm, *delfilter_libxsmm;
  float *bias_libxsmm, *delbias_libxsmm;
  unsigned char *relumask_libxsmm;
  my_eltwise_fuse my_fuse = MY_ELTWISE_FUSE_NONE;
  my_fc_fwd_config my_fc_fwd;
  my_fc_bwd_config my_fc_bwd;

  naive_fullyconnected_t naive_param;
  void* scratch = 0;;

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
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIFm*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOFm*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIFm*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOFm*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Filter     : %10.2f MiB\n", (double)(nIFm*nOFm*sizeof(float))/(1024.0*1024.0) );

  /* allocate data */
  naive_input                = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_delinput             = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_output               = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_deloutput            = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_deloutput_copy       = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_filter               = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_delfilter            = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_bias                 = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);
  naive_delbias              = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);

  naive_libxsmm_delinput     = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_libxsmm_output       = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_libxsmm_delfilter    = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);

  input_libxsmm              = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  delinput_libxsmm           = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  output_libxsmm             = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  deloutput_libxsmm          = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  filter_libxsmm             = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  delfilter_libxsmm          = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  bias_libxsmm               = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);
  delbias_libxsmm            = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);
  relumask_libxsmm           = (unsigned char*)libxsmm_aligned_malloc(((nImg*nOFm)/8)*sizeof(unsigned char), 2097152);

  /* initialize data */
  init_buf( naive_input,     nImg*nIFm, 0, 0 );
  init_buf( naive_delinput,  nImg*nIFm, 0, 0 );
  init_buf( naive_output,    nImg*nOFm, 0, 0 );
  init_buf( naive_deloutput, nImg*nOFm, 0, 0 );
  init_buf( naive_filter,    nIFm*nOFm, 0, 0 );
  init_buf( naive_delfilter, nIFm*nOFm, 0, 0 );
  init_buf( naive_bias,      nOFm,      0, 0 );
  init_buf( naive_delbias,   nOFm,      0, 0 );
  copy_buf( naive_deloutput, naive_deloutput_copy, nImg*nOFm );

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
  matrix_copy_NC_to_NCNC( naive_input,          input_libxsmm,     1, nImg, nIFm, bn, bc );
  matrix_copy_NC_to_NCNC( naive_delinput,       delinput_libxsmm,  1, nImg, nIFm, bn, bc );
  matrix_copy_NC_to_NCNC( naive_output,         output_libxsmm,    1, nImg, nOFm, bn, bk );
  matrix_copy_NC_to_NCNC( naive_deloutput_copy, deloutput_libxsmm, 1, nImg, nOFm, bn, bk );
  matrix_copy_KC_to_KCCK( naive_filter,         filter_libxsmm      , nIFm, nOFm, bc, bk );
  matrix_copy_KC_to_KCCK( naive_delfilter,      delfilter_libxsmm   , nIFm, nOFm, bc, bk );
  copy_buf(naive_bias,    bias_libxsmm,    nOFm);
  copy_buf(naive_delbias, delbias_libxsmm, nOFm);

  /* let's allocate and bind scratch */
  if ( my_fc_fwd.scratch_size > 0 || my_fc_bwd.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( my_fc_fwd.scratch_size, my_fc_bwd.scratch_size);
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
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

    matrix_copy_NCNC_to_NC( output_libxsmm, naive_libxsmm_output, 1, nImg, nOFm, bn, bk );

    /* compare */
    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nOFm, 1, naive_output, naive_libxsmm_output, 0, 0);
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

    matrix_copy_NCNC_to_NC( delinput_libxsmm, naive_libxsmm_delinput, 1, nImg, nIFm, bn, bc );
    matrix_copy_KCCK_to_KC( delfilter_libxsmm, naive_libxsmm_delfilter, nIFm, nOFm, bc, bk );

    /* compare */
    libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nImg*nIFm, 1, naive_delinput, naive_libxsmm_delinput, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_bwd);
    if ( (fuse_type == 1) || (fuse_type == 4) ) {
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nOFm, 1, naive_delbias, delbias_libxsmm, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
    }
    libxsmm_matdiff(&norms_upd, LIBXSMM_DATATYPE_F32, nIFm*nOFm, 1, naive_delfilter, naive_libxsmm_delfilter, 0, 0);
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
  libxsmm_free(naive_deloutput_copy);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_delfilter);
  libxsmm_free(naive_bias);
  libxsmm_free(naive_delbias);
  libxsmm_free(naive_libxsmm_output);
  libxsmm_free(naive_libxsmm_delinput);
  libxsmm_free(naive_libxsmm_delfilter);
  libxsmm_free(input_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(delinput_libxsmm);
  libxsmm_free(deloutput_libxsmm);
  libxsmm_free(filter_libxsmm);
  libxsmm_free(delfilter_libxsmm);
  libxsmm_free(bias_libxsmm);
  libxsmm_free(delbias_libxsmm);
  libxsmm_free(relumask_libxsmm);

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

