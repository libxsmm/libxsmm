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

#include <numa.h>

#define CHECK_L1

/* include c-based dnn library */
#include "../common/dnn_common.h"

LIBXSMM_INLINE void my_init_buf(float* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rng_f64() : (0.05 - libxsmm_rng_f64()/10.0)));
  }
}

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

typedef struct my_opt_config {
  libxsmm_blasint C;
  libxsmm_blasint K;
  libxsmm_blasint bc;
  libxsmm_blasint bk;
  libxsmm_blasint threads;
  float           lr;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
} my_opt_config;

typedef struct my_smax_fwd_config {
  libxsmm_blasint N;
  libxsmm_blasint C;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint threads;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
} my_smax_fwd_config;

typedef struct my_smax_bwd_config {
  libxsmm_blasint N;
  libxsmm_blasint C;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint threads;
  size_t          scratch_size;
  float           loss_weight;
  libxsmm_barrier* barrier;
} my_smax_bwd_config;

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
  libxsmm_blasint fwd_col_teams;
  libxsmm_blasint fwd_row_teams;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
  libxsmm_smmfunction_reducebatch_strd gemm_fwd;
  libxsmm_smmfunction_reducebatch_strd gemm_fwd2;
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
  libxsmm_blasint upd_bf;
  libxsmm_blasint upd_2d_blocking;
  libxsmm_blasint upd_col_teams;
  libxsmm_blasint upd_row_teams;
  libxsmm_blasint ifm_subtasks;
  libxsmm_blasint ofm_subtasks;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
  libxsmm_smmfunction_reducebatch_strd gemm_bwd;
  libxsmm_smmfunction_reducebatch_strd gemm_bwd2;
  libxsmm_smmfunction_reducebatch_strd gemm_upd;
  libxsmm_smmfunction_reducebatch_strd gemm_upd2;
  libxsmm_meltwfunction_unary norm_to_normT_kernel;
} my_fc_bwd_config;

typedef struct my_numa_thr_cfg {
    int thr_s;
    int thr_e;

    int *blocksOFm_s;
    int *blocksOFm_e;
    int *blocksIFm_s;
    int *blocksIFm_e;

    int *blocksOFm_tr_s;
    int *blocksOFm_tr_e;
    int *blocksIFm_tr_s;
    int *blocksIFm_tr_e;

    float **scratch;
    size_t *layer_size;
    int **fwd_ofm_to_numa;

    float *bwd_d_scratch;
    size_t bwd_d_scratch_size;

    float *bwd_w_scratch;
    size_t bwd_w_layer_size;
} my_numa_thr_cfg;

my_fc_fwd_config setup_my_fc_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bn,
                                 libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, my_eltwise_fuse fuse_type) {
  my_fc_fwd_config res;
  libxsmm_blasint lda = bk;
  libxsmm_blasint ldb = bc;
  libxsmm_blasint ldc = bk;
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
  if (threads == 16) {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 2;
    res.fwd_row_teams = 8;
  } else {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 1;
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
  float alpha = 1.0f;
  float beta = 1.0f;
  float zerobeta = 0.0f;
  int updflags = LIBXSMM_GEMM_FLAGS( 'N', 'T' );
  libxsmm_blasint updM;
  libxsmm_blasint updN;

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
    res.bwd_col_teams = 2;
    res.bwd_row_teams = 8;
    res.upd_bf = 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  } else {
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

  /* init scratch */
  res.scratch_size =  sizeof(float) * ( (((size_t)res.C + (size_t)res.K) * (size_t)res.N) + ((size_t)res.C * (size_t)res.K) );

  return res;
}

my_opt_config setup_my_opt(libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bc, libxsmm_blasint bk,
                           libxsmm_blasint threads, float lr) {
  my_opt_config res;

  /* setting up some handle values */
  res.C = C;
  res.K = K;
  res.bc = bc;
  res.bk = bk;
  res.threads = threads;
  res.lr = lr;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* init scratch */
  res.scratch_size = 0;

  return res;
}

my_smax_fwd_config setup_my_smax_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint bn, libxsmm_blasint bc,
                                     libxsmm_blasint threads) {
  my_smax_fwd_config res;

  /* setting up some handle values */
  res.C = C;
  res.N = N;
  res.bc = bc;
  res.bn = bn;
  res.threads = threads;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* init scratch */
  res.scratch_size = 0;

  return res;
}

my_smax_bwd_config setup_my_smax_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint bn, libxsmm_blasint bc,
                                     libxsmm_blasint threads, float loss_weight) {
  my_smax_bwd_config res;

  /* setting up some handle values */
  res.C = C;
  res.N = N;
  res.bc = bc;
  res.bn = bn;
  res.threads = threads;
  res.loss_weight = loss_weight;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* init scratch */
  res.scratch_size = 0;

  return res;
}

void my_fc_fwd_exec( my_fc_fwd_config cfg, const float* in_act_ptr, float* out_act_ptr,
                     const float* bias_ptr, unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch,  my_numa_thr_cfg *numa_thr_cfg, int layer) {
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
  libxsmm_blasint mb1ofm1 = 0, mb1 = 0, ofm1 = 0, ifm1 = 0, mb2 = 0, ofm2 = 0;
  libxsmm_blasint N_tasks_per_thread = 0, M_tasks_per_thread = 0;
  libxsmm_blasint my_M_start = 0, my_M_end = 0, my_N_start = 0, my_N_end = 0;
  libxsmm_blasint my_col_id = 0, my_row_id = 0, col_teams = 0, row_teams = 0;

  LIBXSMM_VLA_DECL(4, float,           output, out_act_ptr, nBlocksOFm, cfg.bn, cfg.bk);
  LIBXSMM_VLA_DECL(4, const float,      input,  in_act_ptr, nBlocksIFm, cfg.bn, cfg.bc);
  LIBXSMM_VLA_DECL(4, const float,     filter, numa_thr_cfg->scratch[layer], nBlocksIFm, cfg.bc, cfg.bk);
  LIBXSMM_VLA_DECL(2, const float,       bias,    bias_ptr,                     cfg.bk);
  LIBXSMM_VLA_DECL(4, unsigned char, relumask,    relu_ptr, nBlocksOFm, cfg.bn, cfg.bk);

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

  const libxsmm_blasint ofm_start = numa_thr_cfg->blocksOFm_s[layer];

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
                for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
                  for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
                    LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = LIBXSMM_VLA_ACCESS(2, bias, ofm1, ofm2, cfg.bk);
                  }
                }
              } else {
                for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
                  for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
                    LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = (float)0;
                  }
                }
              }
            }
            /* BRGEMM */
            cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
                &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
            /* apply post BRGEMM fusion */
            if ( ifm1 == BF-1  ) {
              if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
                for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
                  for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
                    float l_cur_out = LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk);
                    LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = (unsigned char)(( l_cur_out > (float)0 ) ? 1 : 0);
                    l_cur_out = (l_cur_out > (float)0) ? l_cur_out : (float)0;
                    LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = l_cur_out;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
        for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
          if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
            for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
              for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
                LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = LIBXSMM_VLA_ACCESS(2, bias, ofm1, ofm2, cfg.bk);
              }
            }
            cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1-ofm_start, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
                &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
          } else {
            cfg.gemm_fwd2( &LIBXSMM_VLA_ACCESS(4, filter, ofm1-ofm_start, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
                &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
          }
          /* post GEMM fusion */
          if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
            for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
              for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
                float l_cur_out = LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk);
                LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = (unsigned char)(( l_cur_out > (float)0 ) ? 1 : 0);
                l_cur_out = ( l_cur_out > (float)0 ) ? l_cur_out : (float)0;
                LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = l_cur_out;
              }
            }
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
              for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
                for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
                  LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = LIBXSMM_VLA_ACCESS(2, bias, ofm1, ofm2, cfg.bk);
                }
              }
            } else {
              for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
                for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
                  LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = (float)0;
                }
              }
            }
          }
          /* BRGEMM */
          cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
          /* post GEMM fusion */
          if ( ifm1 == BF-1  ) {
            if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
              for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
                for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
                  float l_cur_out = LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk);
                  LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = (unsigned char)(( l_cur_out > (float)0 ) ? 1 : 0);
                  l_cur_out = (l_cur_out > (float)0) ? l_cur_out : (float)0;
                  LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = l_cur_out;
                }
              }
            }
          }
        }
      }
    } else {
      for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
        mb1  = mb1ofm1%nBlocksMB;
        ofm1 = mb1ofm1/nBlocksMB;
        if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
          for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
            for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
              LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = LIBXSMM_VLA_ACCESS(2, bias, ofm1, ofm2, cfg.bk);
            }
          }
          cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(4, filter, ofm1-ofm_start, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
        } else {
          cfg.gemm_fwd2( &LIBXSMM_VLA_ACCESS(4, filter, ofm1-ofm_start, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);
        }
        /* post GEMM fusion */
        if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
          for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
            for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
              float l_cur_out = LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk);
              LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = (unsigned char)(( l_cur_out > (float)0 ) ? 1 : 0);
              l_cur_out = ( l_cur_out > (float)0 ) ? l_cur_out : (float)0;
              LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = l_cur_out;
            }
          }
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

void my_fc_bwd_d_transpose( my_fc_bwd_config cfg, int my_tid, my_numa_thr_cfg **numa_thr_cfg_, int numa_node, int layer, int *ofm_to_node) {
    my_numa_thr_cfg *numa_thr_cfg = *numa_thr_cfg_;
    /* here we assume that input and output blocking is similar */
    const libxsmm_blasint bk = cfg.bk;
    const libxsmm_blasint bc = cfg.bc;
    const libxsmm_blasint nBlocksIFm = cfg.C / bc;
    const libxsmm_blasint nBlocksOFm = cfg.K / bk;

    /* computing first logical thread */
    const libxsmm_blasint ltid = my_tid - numa_thr_cfg[numa_node].thr_s;

    const libxsmm_blasint l_nBlocksIFm = (numa_thr_cfg[numa_node].blocksIFm_tr_e[layer] - numa_thr_cfg[numa_node].blocksIFm_tr_s[layer]) + 1;
    /* number of tasks for transpose that could be run in parallel */
    const libxsmm_blasint transpose_work = l_nBlocksIFm * nBlocksOFm;

    /* compute chunk size */
    int thr = numa_thr_cfg[numa_node].thr_e - numa_thr_cfg[numa_node].thr_s;
    const libxsmm_blasint transpose_chunksize = (transpose_work % thr == 0) ? (transpose_work / thr) : ((transpose_work / thr) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
    const libxsmm_blasint transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

    float *filter_tr = numa_thr_cfg[numa_node].bwd_d_scratch;
    libxsmm_meltw_unary_param trans_param;

    /* lazy barrier init */
    libxsmm_barrier_init(cfg.barrier, my_tid);

    /* transpose weight */
    int ifm1ofm1 = 0;
    for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
      const unsigned int ubk = (unsigned int)bk;
      const unsigned int ubc = (unsigned int)bc;
      int ofm1 = ifm1ofm1 / l_nBlocksIFm;
      int ifm1 = ifm1ofm1 % l_nBlocksIFm;

      my_numa_thr_cfg *l_numa_thr_cfg = &numa_thr_cfg[ofm_to_node[ofm1]];
      float *inp = l_numa_thr_cfg->scratch[layer];
      inp = inp + (ofm1 - l_numa_thr_cfg->blocksOFm_s[layer]) * nBlocksIFm * bc * bk
                  + (ifm1 + numa_thr_cfg[numa_node].blocksIFm_tr_s[layer]) * bc * bk;
      float *out = filter_tr + ifm1 * nBlocksOFm * bk * bc + ofm1 * bk * bc;
      trans_param.in.primary  = (void*)inp;
      trans_param.out.primary = out;
      cfg.norm_to_normT_kernel(&trans_param);
    }

    libxsmm_barrier_wait(cfg.barrier, my_tid);
}

void my_fc_bwd_exec( my_fc_bwd_config cfg, float* din_act_ptr,
                     float* dout_act_ptr, float* dwt_ptr, const float* in_act_ptr,
                     float* dbias_ptr, const unsigned char* relu_ptr, my_pass pass, int start_tid, int my_tid, void* scratch, my_numa_thr_cfg *numa_thr_cfg, int layer ) {
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
  libxsmm_blasint ofm1 = 0, mb1 = 0, ofm2 = 0, mb2 = 0;

  float *grad_output_ptr = (((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) ? ((float*)scratch)+(cfg.C*cfg.K) : dout_act_ptr);
  LIBXSMM_VLA_DECL(4, const float, doutput_orig,    dout_act_ptr, nBlocksOFm, bn, bk);
  LIBXSMM_VLA_DECL(4,       float,      doutput, grad_output_ptr, nBlocksOFm, bn, bk);

  LIBXSMM_VLA_DECL(2,               float,    dbias, dbias_ptr,                     cfg.bk);
  LIBXSMM_VLA_DECL(4, const unsigned char, relumask,  relu_ptr, nBlocksOFm, cfg.bn, cfg.bk);

  const libxsmm_blasint ifm_start = numa_thr_cfg->blocksIFm_tr_s[layer];

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
    for ( mb1ofm1 = eltwise_thr_begin; mb1ofm1 < eltwise_thr_end; ++mb1ofm1 ) {
      mb1  = mb1ofm1%nBlocksMB;
      ofm1 = mb1ofm1/nBlocksMB;

      for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
        for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
          float l_cur_out = LIBXSMM_VLA_ACCESS(4, doutput_orig, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk);
          l_cur_out = (LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) != 0) ? l_cur_out : (float)0;
          LIBXSMM_VLA_ACCESS(4, doutput, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = l_cur_out;
        }
      }
    }

    /* wait for eltwise to finish */
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
    for ( ofm1 = dbias_thr_begin; ofm1 < dbias_thr_end; ++ofm1 ) {
      for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
        LIBXSMM_VLA_ACCESS( 2, dbias, ofm1, ofm2, cfg.bk ) = 0.0f;
      }

      for ( mb1 = 0; mb1 < nBlocksMB; ++mb1 ) {
        for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
          for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
            LIBXSMM_VLA_ACCESS( 2, dbias, ofm1, ofm2, cfg.bk ) += LIBXSMM_VLA_ACCESS(4,  doutput, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk);
          }
        }
      }
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

    /* loop variables */
    libxsmm_blasint ifm1 = 0, ifm2 = 0, mb1ifm1 = 0;
    libxsmm_blasint N_tasks_per_thread = 0, M_tasks_per_thread = 0, my_M_start = 0, my_M_end = 0, my_N_start = 0, my_N_end = 0, my_col_id = 0, my_row_id = 0, col_teams = 0, row_teams = 0;

    LIBXSMM_VLA_DECL(4,       float,    dinput,                 din_act_ptr, nBlocksIFm, bn, bc);
    LIBXSMM_VLA_DECL(4,       float, filter_tr, numa_thr_cfg->bwd_d_scratch, nBlocksOFm, bk, bc);

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

    if (use_2d_blocking == 1) {
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for (ifm1 = my_M_start; ifm1 < my_M_end; ++ifm1) {
            for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
              /* Initialize intermediate f32 tensor */
              if ( ofm1 == 0 ) {
                for ( mb2 = 0; mb2 < bn; ++mb2 ) {
                  for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
                    LIBXSMM_VLA_ACCESS(4, dinput, mb1, ifm1, mb2, ifm2, nBlocksIFm, bn, bc) = (float)0;
                  }
                }
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
              for ( mb2 = 0; mb2 < bn; ++mb2 ) {
                for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
                  LIBXSMM_VLA_ACCESS(4, dinput, mb1, ifm1, mb2, ifm2, nBlocksIFm, bn, bc) = (float)0;
                }
             }
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
          cfg.gemm_bwd2( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1 - ifm_start, 0, 0, 0, nBlocksOFm, bk, bc ),
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
    libxsmm_blasint ifm1ofm1 = 0, ifm1 = 0, ifm2 = 0, bfn = 0, ii = 0, jj = 0;

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
                for (ii = 0; ii<bc; ii++) {
                  for (jj = 0; jj<bk; jj++) {
                    LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ii, jj, nBlocksIFm, bc, bk) = (float)0;
                  }
                }
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
              for (ii = 0; ii<bbc; ii++) {
                for (jj = 0; jj<bbk; jj++) {
                  LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc+ii, ofm2*bbk+jj, nBlocksIFm, bc, bk) = (float)0;
                }
              }
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

void my_opt_exec( my_opt_config cfg, const float* delwt_ptr, int start_tid, int my_tid,
                    my_numa_thr_cfg *numa_thr_cfg, int l, my_fc_fwd_config my_fc_fwd) {
    const libxsmm_blasint ltid = my_tid - numa_thr_cfg->thr_s;

    const libxsmm_blasint nBlocksIFm = my_fc_fwd.C / my_fc_fwd.bc;
    const libxsmm_blasint IFM_shift = my_fc_fwd.bc * my_fc_fwd.bk;
    const libxsmm_blasint OFM_shift = nBlocksIFm *  my_fc_fwd.bc * my_fc_fwd.bk;

    const libxsmm_blasint work = ((numa_thr_cfg->blocksOFm_e[l] - numa_thr_cfg->blocksOFm_s[l]) + 1) * nBlocksIFm;
    /* compute chunk size */
    int thr = numa_thr_cfg->thr_e - numa_thr_cfg->thr_s;
    const libxsmm_blasint chunksize = (work % thr == 0) ? (work / thr) : ((work / thr) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

    libxsmm_barrier_init( cfg.barrier, my_tid );

  __m512 vlr = _mm512_set1_ps( cfg.lr );

    float *dw_prt = (float*)delwt_ptr + numa_thr_cfg->blocksOFm_s[l] * OFM_shift;
    int j = 0, i = 0;
    for (j = thr_begin; j < thr_end; j++) {
        int ofm = j / nBlocksIFm;
        int ifm = j % nBlocksIFm;
        float *out = numa_thr_cfg->scratch[l] + ofm * OFM_shift + ifm * IFM_shift;
        float *inp = dw_prt + ofm * OFM_shift + ifm * IFM_shift;
        for (i = 0; i < IFM_shift; i += 16)
            _mm512_storeu_ps( out+i, _mm512_sub_ps( _mm512_loadu_ps( out+i ), _mm512_mul_ps( vlr, _mm512_loadu_ps( inp + i ) ) ) ) ;

    }

    libxsmm_barrier_wait( cfg.barrier, my_tid );
}

void my_smax_fwd_exec( my_smax_fwd_config cfg, const float* in_act_ptr, float* out_act_ptr, const int* label_ptr, float* loss, int start_tid, int my_tid, void* scratch ) {
  libxsmm_blasint bn = cfg.bn;
  libxsmm_blasint Bn = cfg.N/cfg.bn;
  libxsmm_blasint bc = cfg.bc;
  libxsmm_blasint Bc = cfg.C/cfg.bc;

  /* loop counters */
  libxsmm_blasint i = 0;
  libxsmm_blasint img1, img2, ifm1, ifm2;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could run in parallel for the batch */
  const libxsmm_blasint n_work = Bn * bn;
  /* compute chunk size */
  const libxsmm_blasint n_chunksize = (n_work % cfg.threads == 0) ? (n_work / cfg.threads) : ((n_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint n_thr_begin = (ltid * n_chunksize < n_work) ? (ltid * n_chunksize) : n_work;
  const libxsmm_blasint n_thr_end = ((ltid + 1) * n_chunksize < n_work) ? ((ltid + 1) * n_chunksize) : n_work;

  LIBXSMM_VLA_DECL(4,       float, output, out_act_ptr, Bc, bn, bc);
  LIBXSMM_VLA_DECL(4, const float,  input,  in_act_ptr, Bc, bn, bc);
  LIBXSMM_VLA_DECL(2,   const int,  label,   label_ptr,         bn);

  /* lazy barrier init */
  libxsmm_barrier_init( cfg.barrier, ltid );

  for ( i = n_thr_begin; i < n_thr_end; ++i ) {
    float max =        FLT_MIN;
    float sum_of_exp = 0.0f;

    img1 = i/bn;
    img2 = i%bn;

    /* set output to input and set compute max per image */
    for ( ifm1 = 0; ifm1 < Bc; ++ifm1 ) {
      for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
        LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) = LIBXSMM_VLA_ACCESS( 4, input, img1, ifm1, img2, ifm2, Bc, bn, bc );
        if ( LIBXSMM_VLA_ACCESS( 4, input, img1, ifm1, img2, ifm2, Bc, bn, bc ) > max ) {
          max = LIBXSMM_VLA_ACCESS( 4, input, img1, ifm1, img2, ifm2, Bc, bn, bc );
        }
      }
    }

    /* sum exp over outputs */
    for ( ifm1 = 0; ifm1 < Bc; ++ifm1 ) {
      for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
        LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) = (float)exp( (double)(LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) - max) );
        sum_of_exp += LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc );
      }
    }

    /* scale output */
    sum_of_exp = 1.0f/sum_of_exp;
    for ( ifm1 = 0; ifm1 < Bc; ++ifm1 ) {
      for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
        LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) = LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) * sum_of_exp;
      }
    }
  }

  libxsmm_barrier_wait( cfg.barrier, ltid );

  /* calculate loss single threaded */
  if ( ltid == 0 ) {
    (*loss) = 0.0f;
    for ( img1 = 0; img1 < Bn; ++img1 ) {
      for ( img2 = 0; img2 <bn; ++img2 ) {
        libxsmm_blasint ifm = (libxsmm_blasint)LIBXSMM_VLA_ACCESS( 2, label, img1, img2, bn );
        libxsmm_blasint ifm1b = ifm/bc;
        libxsmm_blasint ifm2b = ifm%bc;
        float val = ( LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1b, img2, ifm2b, Bc, bn, bc ) > FLT_MIN ) ? LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1b, img2, ifm2b, Bc, bn, bc ) : FLT_MIN;
        *loss = LIBXSMM_LOGF( val );
      }
    }
    *loss = ((-1.0f)*(*loss))/cfg.N;
  }

  libxsmm_barrier_wait( cfg.barrier, ltid );
}

void my_smax_bwd_exec( my_smax_bwd_config cfg, float* delin_act_ptr, const float* out_act_ptr, const int* label_ptr, int start_tid, int my_tid, void* scratch ) {
  libxsmm_blasint bn = cfg.bn;
  libxsmm_blasint Bn = cfg.N/cfg.bn;
  libxsmm_blasint bc = cfg.bc;
  libxsmm_blasint Bc = cfg.C/cfg.bc;

  /* loop counters */
  libxsmm_blasint i = 0;
  libxsmm_blasint img1, img2, ifm1, ifm2;

  float rcp_N = 1.0f/cfg.N;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
  /* number of tasks that could run in parallel for the batch */
  const libxsmm_blasint n_work = Bn * bn;
  /* compute chunk size */
  const libxsmm_blasint n_chunksize = (n_work % cfg.threads == 0) ? (n_work / cfg.threads) : ((n_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint n_thr_begin = (ltid * n_chunksize < n_work) ? (ltid * n_chunksize) : n_work;
  const libxsmm_blasint n_thr_end = ((ltid + 1) * n_chunksize < n_work) ? ((ltid + 1) * n_chunksize) : n_work;

  LIBXSMM_VLA_DECL(4, const float, output,   out_act_ptr,  Bc, bn, bc);
  LIBXSMM_VLA_DECL(4,       float, dinput, delin_act_ptr,  Bc, bn, bc);
  LIBXSMM_VLA_DECL(2,   const int,  label,     label_ptr,          bn);

  /* lazy barrier init */
  libxsmm_barrier_init( cfg.barrier, ltid );

  for ( i = n_thr_begin; i < n_thr_end; ++i ) {
    img1 = i/bn;
    img2 = i%bn;

    /* set output to input and set compute max per image */
    for ( ifm1 = 0; ifm1 < Bc; ++ifm1 ) {
      for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
        if ( (ifm1*Bc)+ifm2 == (libxsmm_blasint)LIBXSMM_VLA_ACCESS( 2, label, img1, img2, bn ) ) {
          LIBXSMM_VLA_ACCESS( 4, dinput, img1, ifm1, img2, ifm2, Bc, bn, bc ) =
            ( LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) - 1.0f ) * rcp_N * cfg.loss_weight;
        } else {
          LIBXSMM_VLA_ACCESS( 4, dinput, img1, ifm1, img2, ifm2, Bc, bn, bc ) =
            LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) * rcp_N * cfg.loss_weight;
        }
      }
    }
  }

  libxsmm_barrier_wait( cfg.barrier, ltid );
}

void *numa_alloc_onnode_aligned(size_t size, int numa_node, int alignment_) {
#if 0
    int alignment = alignment_ - 1;
    size_t adj_size = sizeof(size_t) + alignment;

    void *r_ptr = NULL;
    void *t_ptr = numa_alloc_onnode(size + adj_size, numa_node);
    if (t_ptr == NULL) return NULL;

    r_ptr = (void *)(((size_t)t_ptr + adj_size) & ~alignment);
    *((size_t*)r_ptr - 1) = (size_t)r_ptr - (size_t)t_ptr;

    return r_ptr;
#else
    return numa_alloc_onnode(size, numa_node);

#endif
}

void numa_free_aligned(void *ptr, size_t size) {
#if 0
    if (ptr == NULL) return;

    void *t_ptr = (void*)((size_t*)ptr - *((size_t*)ptr - 1));
    numa_free(t_ptr, size);
#else
    numa_free(ptr, size);
#endif
}

int setup_my_numa(my_numa_thr_cfg **numa_thr_cfg_, int num_layers, int n_threads) {
    int max_nodes = numa_max_node() + 1;
    int max_cfg_nodes = numa_num_configured_nodes();
    int max_cfg_cpus = numa_num_configured_cpus();

    int max_task_cpus = numa_num_task_cpus();


    my_numa_thr_cfg *numa_thr_cfg = (my_numa_thr_cfg *) malloc(sizeof(my_numa_thr_cfg) * max_cfg_nodes);

    printf("NUMA configuration:\n");
    printf("There are %d numa nodes on the system\n", max_nodes);
    printf("There are %d configured numa nodes on the system\n", max_cfg_nodes);
    printf("There are %d configured CPUs on the system\n", max_cfg_cpus);
    printf("There are %d CPUs asigned for the current task\n", max_task_cpus);

    struct bitmask* bmask = numa_bitmask_alloc(max_cfg_cpus);
    int thr_count = 0, i = 0;
    for (i = 0; i < max_cfg_nodes; i++) {
        numa_node_to_cpus(i, bmask);

        numa_thr_cfg[i].scratch = (float**) malloc(sizeof(float*) * num_layers);
        numa_thr_cfg[i].layer_size = (size_t*)malloc(sizeof(size_t)*num_layers);

        numa_thr_cfg[i].blocksOFm_s = (int*)malloc(sizeof(int)*num_layers);
        numa_thr_cfg[i].blocksOFm_e = (int*)malloc(sizeof(int)*num_layers);
        numa_thr_cfg[i].blocksIFm_s = (int*)malloc(sizeof(int)*num_layers);
        numa_thr_cfg[i].blocksIFm_e = (int*)malloc(sizeof(int)*num_layers);

        numa_thr_cfg[i].blocksOFm_tr_s = (int*)malloc(sizeof(int)*num_layers);
        numa_thr_cfg[i].blocksOFm_tr_e = (int*)malloc(sizeof(int)*num_layers);
        numa_thr_cfg[i].blocksIFm_tr_s = (int*)malloc(sizeof(int)*num_layers);
        numa_thr_cfg[i].blocksIFm_tr_e = (int*)malloc(sizeof(int)*num_layers);
        /*
            printf("@@@@@ node %d size %zd cpus ", i, bmask->size);
            size_t j = 0;
            for(j = 0; j < bmask->size; j++)
            printf("%d", numa_bitmask_isbitset(bmask, j));
            printf("\n");
        */
        int num_threads_in_mask = 0;
        int t = 0;
        for (t = 0; t < bmask->size; t++)
        if (numa_bitmask_isbitset(bmask, t)) num_threads_in_mask++;

        int node_threads = 0;
        while(thr_count < n_threads && node_threads < num_threads_in_mask) {
            if (numa_bitmask_isbitset(bmask, thr_count)) {
                numa_thr_cfg[i].thr_s = thr_count;
                break;
            }
            thr_count++; node_threads++;
        }
        while(thr_count < n_threads && node_threads < num_threads_in_mask) {
            if (numa_bitmask_isbitset(bmask, thr_count))
                numa_thr_cfg[i].thr_e = thr_count;
            thr_count++; node_threads++;
        }
    }

    *numa_thr_cfg_ = numa_thr_cfg;

    return 1;
}

int setup_my_numa_fwd(my_numa_thr_cfg **numa_thr_cfg_, int num_layers, my_fc_fwd_config* my_fc_fwd) {
    my_numa_thr_cfg *numa_thr_cfg = *numa_thr_cfg_;

    int max_cfg_nodes = numa_num_configured_nodes();
    int i = 0;
    for (i = 0; i < max_cfg_nodes; i++) {
        int l = 0;
        for (l = 0; l < num_layers; l++) {
            const libxsmm_blasint nBlocksOFm = my_fc_fwd[l].K / my_fc_fwd[l].bk;
            const libxsmm_blasint nBlocksMB  = my_fc_fwd[l].N / my_fc_fwd[l].bn;
            if (my_fc_fwd[l].fwd_bf > 1) {
                printf("@@@ NUMA ERROR: doesn't support this configuration\n");
                return -1;
            }
            int thr = 0;
            if (my_fc_fwd[l].fwd_2d_blocking == 1) {
                libxsmm_blasint row_teams = my_fc_fwd[l].fwd_row_teams;
                libxsmm_blasint M_tasks_per_thread = LIBXSMM_UPDIV(nBlocksOFm, row_teams);

                numa_thr_cfg[i].blocksOFm_s[l] = nBlocksOFm;
                numa_thr_cfg[i].blocksOFm_e[l] = 0;
                for (thr = numa_thr_cfg[i].thr_s; thr <= numa_thr_cfg[i].thr_e; thr++) {
                    libxsmm_blasint my_row_id = thr % row_teams; /* ltid */

                    libxsmm_blasint my_M_start = LIBXSMM_MIN(my_row_id * M_tasks_per_thread, nBlocksOFm);
                    libxsmm_blasint my_M_end = LIBXSMM_MIN((my_row_id+1) * M_tasks_per_thread, nBlocksOFm);

                    numa_thr_cfg[i].blocksOFm_s[l] = (my_M_start < numa_thr_cfg[i].blocksOFm_s[l])
                        ? my_M_start
                        : numa_thr_cfg[i].blocksOFm_s[l];

                    numa_thr_cfg[i].blocksOFm_e[l] = (my_M_end > numa_thr_cfg[i].blocksOFm_e[l])
                        ? my_M_end
                        : numa_thr_cfg[i].blocksOFm_e[l];
                }
            } else {
                numa_thr_cfg[i].blocksOFm_s[l] = nBlocksOFm;
                numa_thr_cfg[i].blocksOFm_e[l] = 0;
                for (thr = numa_thr_cfg[i].thr_s; thr <= numa_thr_cfg[i].thr_e; thr++) {
                    const libxsmm_blasint work = nBlocksOFm * nBlocksMB;
                    const libxsmm_blasint chunksize = (work % my_fc_fwd[l].threads == 0) ?
                            (work / my_fc_fwd[l].threads) : ((work / my_fc_fwd[l].threads) + 1);
                    const libxsmm_blasint thr_begin = (thr * chunksize < work) ? (thr * chunksize) : work;
                    const libxsmm_blasint thr_end = ((thr + 1) * chunksize < work) ? ((thr + 1) * chunksize) : work;

                    int ofm_s = thr_begin / nBlocksMB;
                    int ofm_e = (thr_end-1) / nBlocksMB;

                    numa_thr_cfg[i].blocksOFm_s[l] = (ofm_s < numa_thr_cfg[i].blocksOFm_s[l])
                        ? ofm_s
                        : numa_thr_cfg[i].blocksOFm_s[l];

                    numa_thr_cfg[i].blocksOFm_e[l] = (ofm_e > numa_thr_cfg[i].blocksOFm_e[l])
                        ? ofm_e
                        : numa_thr_cfg[i].blocksOFm_e[l];
                }
                #if 0
                printf("numa_thr_cfg[%d].blocksOFm_s[%d] %d numa_thr_cfg[%d].blocksOFm_e[%d] %d\n",
                    i, l, numa_thr_cfg[i].blocksOFm_s[l], i, l, numa_thr_cfg[i].blocksOFm_e[l]);
                #endif
            }
        }
    }
    return 1;
}

void set_fwd_ofm_to_node(int **fwd_ofm_to_node, my_numa_thr_cfg **numa_thr_cfg_, int num_layers, my_fc_fwd_config* my_fc_fwd) {
    int max_cfg_nodes = numa_num_configured_nodes();
    my_numa_thr_cfg *numa_thr_cfg = *numa_thr_cfg_;

    int l, ofm, i;
    for (l = 0; l < num_layers; l++) {
        const libxsmm_blasint nBlocksOFm = my_fc_fwd[l].K / my_fc_fwd[l].bk;
        fwd_ofm_to_node[l] = (int*) malloc(sizeof(int) * nBlocksOFm);
        int *l_fwd_ofm_to_node = fwd_ofm_to_node[l];
        for (i = 0; i < max_cfg_nodes; i++) {
            for (ofm = 0; ofm < nBlocksOFm; ofm++) {
                if (ofm >= numa_thr_cfg[i].blocksOFm_s[l] && ofm <= numa_thr_cfg[i].blocksOFm_e[l])
                    l_fwd_ofm_to_node[ofm] = i;
            }
        }
    }
#if 0
    for (l = 0; l < num_layers; l++) {
        const libxsmm_blasint nBlocksOFm = my_fc_fwd[l].K / my_fc_fwd[l].bk;
        int *l_fwd_ofm_to_node = fwd_ofm_to_node[l];
        for (ofm = 0; ofm < nBlocksOFm; ofm++)
            printf("%d l_fwd_ofm_to_node[%d] %d | %d\n", l, ofm, l_fwd_ofm_to_node[ofm], nBlocksOFm);
    }
#endif
}

void free_fwd_ofm_to_node(int **fwd_ofm_to_node, int num_layers) {
    int l;
    for (l = 0; l < num_layers; l++) {
        free(fwd_ofm_to_node[l]);
    }
}

int setup_my_numa_bwd_d(my_numa_thr_cfg **numa_thr_cfg_, int num_layers, my_fc_bwd_config* my_fc_bwd) {
    my_numa_thr_cfg *numa_thr_cfg = *numa_thr_cfg_;

    int max_cfg_nodes = numa_num_configured_nodes();
    int i = 0;
    for (i = 0; i < max_cfg_nodes; i++) {
        int l = 0;
        for (l = 0; l < num_layers; l++) {
            if (my_fc_bwd[l].bwd_bf > 1) {
                printf("@@@ NUMA ERROR: doesn't support this configuration\n");
                return -1;
            }
            int thr = 0;
            const libxsmm_blasint nBlocksIFm = my_fc_bwd[l].C / my_fc_bwd[l].bc;
            const libxsmm_blasint nBlocksMB  = my_fc_bwd[l].N / my_fc_bwd[l].bn;

            if (my_fc_bwd[l].bwd_2d_blocking == 1) {
                printf("@@@ NUMA ERROR: doesn't support this configuration\n");
                return -1;
            } else {
                numa_thr_cfg[i].blocksIFm_tr_s[l] = nBlocksIFm;
                numa_thr_cfg[i].blocksIFm_tr_e[l] = 0;
                for (thr = numa_thr_cfg[i].thr_s; thr <= numa_thr_cfg[i].thr_e; thr++) {
                    /* number of tasks that could be run in parallel */
                    const libxsmm_blasint work = nBlocksIFm * nBlocksMB;
                    /* compute chunk size */
                    const libxsmm_blasint chunksize = (work % my_fc_bwd[l].threads == 0) ?
                            (work / my_fc_bwd[l].threads) : ((work / my_fc_bwd[l].threads) + 1);
                    /* compute thr_begin and thr_end */
                    const libxsmm_blasint thr_begin = (thr * chunksize < work) ? (thr * chunksize) : work;
                    const libxsmm_blasint thr_end = ((thr + 1) * chunksize < work) ? ((thr + 1) * chunksize) : work;

                    int ifm_s = thr_begin / nBlocksMB;
                    int ifm_e = (thr_end-1) / nBlocksMB;

                    numa_thr_cfg[i].blocksIFm_tr_s[l] = (ifm_s < numa_thr_cfg[i].blocksIFm_tr_s[l])
                        ? ifm_s
                        : numa_thr_cfg[i].blocksIFm_tr_s[l];

                    numa_thr_cfg[i].blocksIFm_tr_e[l] = (ifm_e > numa_thr_cfg[i].blocksIFm_tr_e[l])
                        ? ifm_e
                        : numa_thr_cfg[i].blocksIFm_tr_e[l];
                }
                #if 0
                printf("numa_thr_cfg[%d].blocksIFm_tr_s[%d] %d numa_thr_cfg[%d].blocksIFm_tr_e[%d] %d\n",
                    i, l, numa_thr_cfg[i].blocksIFm_tr_s[l], i, l, numa_thr_cfg[i].blocksIFm_tr_e[l]);
                #endif
            }
        }
    }
    return 1;
}

int allocate_numa_buffers_fwd(my_numa_thr_cfg **numa_thr_cfg_, int num_layers, my_fc_fwd_config* my_fc_fwd) {
     my_numa_thr_cfg *numa_thr_cfg = *numa_thr_cfg_;

    int max_cfg_nodes = numa_num_configured_nodes();
    int i = 0, l = 0;
    for (i = 0; i < max_cfg_nodes; i++) {
        for (l = 0; l < num_layers; l++) {
            const libxsmm_blasint nBlocksIFm = my_fc_fwd[l].C / my_fc_fwd[l].bc;
            const libxsmm_blasint OFM_shift = nBlocksIFm *  my_fc_fwd[l].bc * my_fc_fwd[l].bk;

            int l_nBlocksOFm = (numa_thr_cfg[i].blocksOFm_e[l] - numa_thr_cfg[i].blocksOFm_s[l]) + 1;
            if (l_nBlocksOFm <= 0)
                continue;
            numa_thr_cfg[i].layer_size[l] = sizeof(float) * ((l_nBlocksOFm) * OFM_shift);
            numa_thr_cfg[i].scratch[l] = (float*)numa_alloc_onnode_aligned(numa_thr_cfg[i].layer_size[l], i, 2097152);
            if (numa_thr_cfg[i].scratch[l] == NULL) {
                printf("@@@ NUMA ERROR: cannot allocate on node #%d\n", i);
                return -1;
            }

        }
    }
    return 1;
}

int allocate_numa_buffers_bwd_d(my_numa_thr_cfg **numa_thr_cfg_, int num_layers, my_fc_bwd_config* my_fc_bwd) {
     my_numa_thr_cfg *numa_thr_cfg = *numa_thr_cfg_;

    int max_cfg_nodes = numa_num_configured_nodes();
    int i = 0, l = 0;
    for (i = 0; i < max_cfg_nodes; i++) {
        int l_nBlocksIFm = 0;
        for (l = 0; l < num_layers; l++) {
            const libxsmm_blasint nBlocksOFm = my_fc_bwd[l].K / my_fc_bwd[l].bk;
            const libxsmm_blasint IFM_shift = nBlocksOFm *  my_fc_bwd[l].bc * my_fc_bwd[l].bk;

            if (l_nBlocksIFm <= ((numa_thr_cfg[i].blocksIFm_tr_e[l] - numa_thr_cfg[i].blocksIFm_tr_s[l]) + 1) * IFM_shift)
                l_nBlocksIFm = ((numa_thr_cfg[i].blocksIFm_tr_e[l] - numa_thr_cfg[i].blocksIFm_tr_s[l]) + 1) * IFM_shift;
        }
        numa_thr_cfg[i].bwd_d_scratch_size = sizeof(float) * (l_nBlocksIFm);
        numa_thr_cfg[i].bwd_d_scratch = (float*)numa_alloc_onnode_aligned(numa_thr_cfg[i].bwd_d_scratch_size, i, 2097152);
        if (numa_thr_cfg[i].bwd_d_scratch == NULL) {
            printf("@@@ NUMA ERROR: cannot allocate on node #%d\n", i);
            return -1;
        }
    }
    return 1;
}

int copy_to_numa_buffers_fwd_inf(my_numa_thr_cfg **numa_thr_cfg_, int num_layers, my_fc_fwd_config* my_fc_fwd, float **fil_libxsmm) {
     my_numa_thr_cfg *numa_thr_cfg = *numa_thr_cfg_;

    int max_cfg_nodes = numa_num_configured_nodes();

    int i,l;
#ifndef COPY_ON_LOCAL_NODES
    #pragma omp parallel for collapse(2) private (i,l)
#else
    #pragma omp parallel private (i,l)
    {
        int tid = omp_get_thread_num();
#endif
        for (i = 0; i < max_cfg_nodes; i++) {
#ifdef COPY_ON_LOCAL_NODES
            if (tid >= numa_thr_cfg[i].thr_s && tid <= numa_thr_cfg[i].thr_e) {
                numa_run_on_node(i);
            }
            if (tid == numa_thr_cfg[i].thr_s) {

#endif
                for (l = 0; l < num_layers; l++) {
                    const libxsmm_blasint nBlocksIFm = my_fc_fwd[l].C / my_fc_fwd[l].bc;
                    const libxsmm_blasint BOFM_shift = nBlocksIFm *  my_fc_fwd[l].bc * my_fc_fwd[l].bk;

                    int l_nBlocksOFm = (numa_thr_cfg[i].blocksOFm_e[l] - numa_thr_cfg[i].blocksOFm_s[l]) + 1;
                    int j = 0;
                    for (j = 0; j < l_nBlocksOFm ; j++) {
                        size_t l_BOFM_shift = j * BOFM_shift;
                        float *out = numa_thr_cfg[i].scratch[l] + l_BOFM_shift;
                        float *inp = fil_libxsmm[l] + numa_thr_cfg[i].blocksOFm_s[l] * BOFM_shift + l_BOFM_shift;
                        memcpy(out, inp, sizeof(float) * nBlocksIFm *  my_fc_fwd[l].bc * my_fc_fwd[l].bk);
                    }
                }
#ifdef COPY_ON_LOCAL_NODES
            }
#endif
        }
#ifdef COPY_ON_LOCAL_NODES
    }
#endif
    return 1;
}

int copy_to_numa_buffers_fwd(my_numa_thr_cfg *numa_thr_cfg, my_fc_fwd_config my_fc_fwd, float *fil_libxsmm, int numa_node, int l, int my_tid, int dir) {
    const libxsmm_blasint ltid = my_tid - numa_thr_cfg->thr_s;

    const libxsmm_blasint nBlocksIFm = my_fc_fwd.C / my_fc_fwd.bc;

    const libxsmm_blasint IFM_shift = my_fc_fwd.bc * my_fc_fwd.bk;
    const libxsmm_blasint OFM_shift = nBlocksIFm *  my_fc_fwd.bc * my_fc_fwd.bk;

    const libxsmm_blasint work = ((numa_thr_cfg->blocksOFm_e[l] - numa_thr_cfg->blocksOFm_s[l]) + 1) * nBlocksIFm;
    /* compute chunk size */
    int thr = numa_thr_cfg->thr_e - numa_thr_cfg->thr_s;
    const libxsmm_blasint chunksize = (work % thr == 0) ? (work / thr) : ((work / thr) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

    /*libxsmm_barrier_init( my_fc_fwd.barrier, my_tid );*/

    float *inp, *out;
    if (dir) {
       inp = numa_thr_cfg->scratch[l];
       out = fil_libxsmm + numa_thr_cfg->blocksOFm_s[l] * OFM_shift;
    } else {
       out = numa_thr_cfg->scratch[l];
       inp = fil_libxsmm + numa_thr_cfg->blocksOFm_s[l] * OFM_shift;
    }

    int j = 0;
    for (j = thr_begin; j < thr_end; j++) {
        int ofm = j / nBlocksIFm;
        int ifm = j % nBlocksIFm;
        float *l_out = out + ofm * OFM_shift + ifm * IFM_shift;
        float *l_inp = inp + ofm * OFM_shift + ifm * IFM_shift;

        memcpy(l_out, l_inp, sizeof(float) * IFM_shift);

    }

    /*libxsmm_barrier_wait( my_fc_fwd.barrier, my_tid );*/

    return 1;
}

int main(int argc, char* argv[])
{
  float **act_libxsmm, **fil_libxsmm, **delact_libxsmm, **delfil_libxsmm;
  float **bias_libxsmm, **delbias_libxsmm;
  unsigned char **relumask_libxsmm;
  int *label_libxsmm;
  my_eltwise_fuse my_fuse;
  my_fc_fwd_config* my_fc_fwd;
  my_fc_bwd_config* my_fc_bwd;
  my_opt_config* my_opt;
  my_smax_fwd_config my_smax_fwd;
  my_smax_bwd_config my_smax_bwd;
  void* scratch = NULL;
  size_t scratch_size = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int MB = 256;          /* mini-batch size, "N" */
  int fuse_type = 0;      /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  int bn = 32;
  int bk = 32;
  int bc = 32;
  int *C;               /* number of input feature maps, "C" */
  int num_layers = 0;

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  unsigned long long *fwd_time, *bwd_time, *solver_time;
  double l_total = 0.0;
  double gflop = 0.0;
  int i, j;
  double fil_size = 0.0;
  double act_size = 0.0;
  float lr = 0.2f;
  float loss_weight = 0.1f;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&norms_upd);
  libxsmm_matdiff_clear(&diff);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters MB fuse_type type bn bk bc C1 C2 ... CN\n", argv[0]);
    return 0;
  }
  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  num_layers = argc - 9;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) MB         = atoi(argv[i++]);
  if (argc > i) fuse_type  = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);
  if (argc > i) bn         = atoi(argv[i++]);
  if (argc > i) bk         = atoi(argv[i++]);
  if (argc > i) bc         = atoi(argv[i++]);
  /* allocate the number of channles buffer */
  if ( num_layers < 1 ) {
    printf("Usage: %s iters MB fuse_type type bn bk bc C1 C2 ... CN\n", argv[0]);
    return 0;
  }
  C = (int*)malloc((num_layers+2)*sizeof(int));
  for (j = 0 ; i < argc; ++i, ++j ) {
    C[j] = atoi(argv[i]);
  }
  /* handle softmax config */
  C[num_layers+1] = C[num_layers];

  if (type != 'A' && type != 'F' && type != 'B') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only)\n");
    return -1;
  }
  if ( (fuse_type < 0) || (fuse_type > 5) ) {
    printf("fuse type needs to be 0 (None), 1 (Bias), 2 (ReLU), 3 (Sigmoid), 4 (Bias+ReLU), 5 (Bias+Sigmoid)\n");
    return -1;
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
  printf("PARAMS: N:%d\n", MB);
  printf("PARAMS: Layers: %d\n", num_layers);
  printf("PARAMS: ITERS:%d", iters); printf("  Threads:%d\n", nThreads);
  for (i = 0; i < num_layers; ++i ) {
    if (i == 0) {
      act_size += (double)(MB*C[i]*sizeof(float))/(1024.0*1024.0);
      printf("SIZE Activations  %i (%dx%d): %10.2f MiB\n", i, MB, C[i], (double)(MB*C[i]*sizeof(float))/(1024.0*1024.0) );
    }
    act_size += (double)(MB*C[i+1]*sizeof(float))/(1024.0*1024.0);
    fil_size += (double)(C[i]*C[i+1]*sizeof(float))/(1024.0*1024.0);
    printf("SIZE Filter       %i (%dx%d): %10.2f MiB\n", i, C[i], C[i+1], (double)(C[i]*C[i+1]*sizeof(float))/(1024.0*1024.0) );
    printf("SIZE Activations  %i (%dx%d): %10.2f MiB\n", i+1, MB, C[i+1], (double)(MB*C[i+1]*sizeof(float))/(1024.0*1024.0) );
  }
  act_size += (double)(MB*C[num_layers+1]*sizeof(float))/(1024.0*1024.0);
  printf("SIZE Activations softmax (%dx%d): %10.2f MiB\n", MB, C[num_layers+1], (double)(MB*C[num_layers+1]*sizeof(float))/(1024.0*1024.0) );
  printf("\nTOTAL SIZE Activations:    %10.2f MiB\n", act_size );
  printf("TOTAL SIZE Filter:         %10.2f MiB\n", fil_size );
  printf("TOTAL SIZE delActivations: %10.2f MiB\n", act_size );
  printf("TOTAL SIZE delFilter:      %10.2f MiB\n", fil_size );
  printf("TOTAL SIZE MLP:            %10.2f MiB\n", (2.0*fil_size) + (2.0*act_size) );

  /* allocate data */
  /* +2 because of the softwax layer */
  act_libxsmm    = (float**)malloc( (num_layers+2)*sizeof(float*) );
  delact_libxsmm = (float**)malloc( (num_layers+1)*sizeof(float*) );
  for ( i = 0 ; i < num_layers+2; ++i ) {
#ifdef ACT_NUMA_INTERLEAVED
    act_libxsmm[i]                = (float*)numa_alloc_interleaved( MB*C[i]*sizeof(float));
#else
    act_libxsmm[i]                = (float*)libxsmm_aligned_malloc( MB*C[i]*sizeof(float), 2097152);
#endif
    /* softmax has no incoming gradients */
    if ( i < num_layers+1 ) {
      delact_libxsmm[i]             = (float*)libxsmm_aligned_malloc( MB*C[i]*sizeof(float), 2097152);
    }
  }
  fil_libxsmm    = (float**)malloc( num_layers*sizeof(float*) );
  delfil_libxsmm = (float**)malloc( num_layers*sizeof(float*) );
  for ( i = 0 ; i < num_layers; ++i ) {
    fil_libxsmm[i]                = (float*)libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(float), 2097152);
    delfil_libxsmm[i]             = (float*)libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(float), 2097152);
  }
  bias_libxsmm    = (float**)malloc( num_layers*sizeof(float*) );
  delbias_libxsmm = (float**)malloc( num_layers*sizeof(float*) );
  for ( i = 0 ; i < num_layers; ++i ) {
    bias_libxsmm[i]               = (float*)libxsmm_aligned_malloc( C[i+1]*sizeof(float), 2097152);
    delbias_libxsmm[i]            = (float*)libxsmm_aligned_malloc( C[i+1]*sizeof(float), 2097152);
  }
  relumask_libxsmm = (unsigned char**)malloc( num_layers*sizeof(unsigned char*) );
  for ( i = 0 ; i < num_layers; ++i ) {
    relumask_libxsmm[i]           = (unsigned char*)libxsmm_aligned_malloc( MB*C[i+1]*sizeof(unsigned char), 2097152);
  }
  label_libxsmm = (int*)libxsmm_aligned_malloc( MB*sizeof(int), 2097152);

  /* init data */
  for ( i = 0 ; i < num_layers+2; ++i ) {
    my_init_buf( act_libxsmm[i], MB*C[i], 0, 0 );
  }
  for ( i = 0 ; i < num_layers+1; ++i ) {
    my_init_buf( delact_libxsmm[i], MB*C[i], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf( fil_libxsmm[i], C[i]*C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf( delfil_libxsmm[i], C[i]*C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf( bias_libxsmm[i], C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf( delbias_libxsmm[i], C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    zero_buf_uint8( relumask_libxsmm[i], MB*C[i+1] );
  }
  zero_buf_int32( label_libxsmm, MB );

  printf("\n");
  printf("##########################################\n");
  printf("#      Setting Up  (custom-Storage)      #\n");
  printf("##########################################\n");

  if ( fuse_type == 0 ) {
    my_fuse = MY_ELTWISE_FUSE_NONE;
  } else if ( fuse_type == 1 ) {
    my_fuse = MY_ELTWISE_FUSE_BIAS;
  } else if ( fuse_type == 2 ) {
    my_fuse = MY_ELTWISE_FUSE_RELU;
  } else if ( fuse_type == 4 ) {
    my_fuse = MY_ELTWISE_FUSE_BIAS_RELU;
  } else {
    my_fuse = MY_ELTWISE_FUSE_NONE;
  }

  /* allocating handles */
  my_fc_fwd = (my_fc_fwd_config*) malloc( num_layers*sizeof(my_fc_fwd_config) );
  my_fc_bwd = (my_fc_bwd_config*) malloc( num_layers*sizeof(my_fc_bwd_config) );
  my_opt    = (my_opt_config*)    malloc( num_layers*sizeof(my_opt_config)    );

  /* setting up handles + scratch */
  for ( i = 0; i < num_layers; ++i ) {
    my_fc_fwd[i] = setup_my_fc_fwd(MB, C[i], C[i+1], (MB % bn == 0) ? bn : MB,
                                             (C[i  ] % bc == 0) ? bc : C[i  ],
                                             (C[i+1] % bk == 0) ? bk : C[i+1],
                                             nThreads, my_fuse);

    my_fc_bwd[i] = setup_my_fc_bwd(MB, C[i], C[i+1], (MB % bn == 0) ? bn : MB,
                                             (C[i  ] % bc == 0) ? bc : C[i  ],
                                             (C[i+1] % bk == 0) ? bk : C[i+1],
                                              nThreads, my_fuse);

    my_opt[i] = setup_my_opt( C[i], C[i+1], (C[i  ] % bc == 0) ? bc : C[i  ],
                                            (C[i+1] % bk == 0) ? bk : C[i+1],
                                            nThreads, lr );

    /* let's allocate and bind scratch */
    if ( my_fc_fwd[i].scratch_size > 0 || my_fc_bwd[i].scratch_size > 0 || my_opt[i].scratch_size > 0 ) {
      size_t alloc_size = LIBXSMM_MAX( LIBXSMM_MAX( my_fc_fwd[i].scratch_size, my_fc_bwd[i].scratch_size), my_opt[i].scratch_size );
      if ( alloc_size > scratch_size ) {
        if ( scratch != NULL ) libxsmm_free( scratch );
        scratch_size = alloc_size;
        scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
        my_init_buf( (float*)(scratch), (scratch_size)/4, 0, 0 );
      }
    }
  }

  /* softmax+loss is treated as N+! layer */
  my_smax_fwd = setup_my_smax_fwd( MB, C[num_layers+1], (MB % bn == 0) ? bn : MB,
                                       (C[num_layers+1] % bk == 0) ? bk : C[num_layers+1],
                                       nThreads );

  my_smax_bwd = setup_my_smax_bwd( MB, C[num_layers+1], (MB % bn == 0) ? bn : MB,
                                       (C[num_layers+1] % bk == 0) ? bk : C[num_layers+1],
                                       nThreads, loss_weight );

  if ( my_smax_fwd.scratch_size > 0 || my_smax_bwd.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( my_smax_fwd.scratch_size, my_smax_bwd.scratch_size );
    if ( alloc_size > scratch_size ) {
      if ( scratch != NULL ) libxsmm_free( scratch );
      scratch_size = alloc_size;
      scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
      my_init_buf( (float*)(scratch), (scratch_size)/4, 0, 0 );
    }
  }

  my_numa_thr_cfg *numa_thr_cfg;
  /* Define numa configuration: #numa nodes, #threads on each node */
  setup_my_numa(&numa_thr_cfg, num_layers, nThreads);

  if ( type == 'F') {
    printf("##########################################\n");
    printf("#   Performance - FWD (custom-Storage)   #\n");
    printf("##########################################\n");

    setup_my_numa_fwd(&numa_thr_cfg, num_layers, my_fc_fwd);
    allocate_numa_buffers_fwd(&numa_thr_cfg, num_layers, my_fc_fwd);
    l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#   pragma omp parallel private(i,j)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      const int numa_node = numa_node_of_cpu(tid);
      for ( i = 0; i < num_layers; ++i) {
        copy_to_numa_buffers_fwd(&numa_thr_cfg[numa_node], my_fc_fwd[i], fil_libxsmm[i], numa_node, i, tid, 0);
      }
      for (j = 0; j < iters; ++j) {
        for ( i = 0; i < num_layers; ++i) {
          my_fc_fwd_exec( my_fc_fwd[i], act_libxsmm[i], act_libxsmm[i+1],
                          bias_libxsmm[i], relumask_libxsmm[i], 0, tid, scratch, &numa_thr_cfg[numa_node], i);
        }
#ifdef USE_SOFTMAX
        my_smax_fwd_exec( my_smax_fwd, act_libxsmm[num_layers], act_libxsmm[num_layers+1], label_libxsmm, &loss,
                          0, tid, scratch );
#endif
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = 0.0;
    for ( i = 0; i < num_layers; ++i) {
      gflop += (2.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);
    printf("PERFDUMP,FP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
    for ( i = 0; i < num_layers; ++i ) {
      printf("%i,", C[i] );
    }
    printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);

    /* Print some norms on last act for fwd and weights of first layer after all iterations */
    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, MB*C[num_layers], 1, act_libxsmm[num_layers], act_libxsmm[num_layers], 0, 0);
    printf("L1 of act[num_layers]  : %.25g\n", norms_fwd.l1_ref);
  }

  if (type == 'B') {
    printf("##########################################\n");
    printf("#   NOT Supported: Performance - BWD (custom-Storage)   #\n");
    printf("##########################################\n");
    exit( -1 );
#if 0
    l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i,j)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      for (j = 0; j < iters; ++j) {
#ifdef USE_SOFTMAX
        my_smax_bwd_exec( my_smax_bwd, delact_libxsmm[num_layers], act_libxsmm[num_layers+1], label_libxsmm,
                          0, tid, scratch );
#endif
        for ( i = num_layers-1; i > 0; --i) {
          my_fc_bwd_exec( my_fc_bwd[i], fil_libxsmm[i], delact_libxsmm[i], delact_libxsmm[i+1], delfil_libxsmm[i],
                          act_libxsmm[i], delbias_libxsmm[i], relumask_libxsmm[i], MY_PASS_BWD, 0, tid, scratch );
          my_opt_exec( my_opt[i], fil_libxsmm[i], delfil_libxsmm[i], 0, tid, scratch );
        }
        my_fc_bwd_exec( my_fc_bwd[0], fil_libxsmm[0], delact_libxsmm[0], delact_libxsmm[0+1], delfil_libxsmm[0],
                        act_libxsmm[0], delbias_libxsmm[0], relumask_libxsmm[0], MY_PASS_BWD_W, 0, tid, scratch );
        my_opt_exec( my_opt[0], fil_libxsmm[0], delfil_libxsmm[0], 0, tid, scratch );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = 0.0;
    for ( i = num_layers-1; i > 0; --i) {
      gflop += (4.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    gflop += (2.0*(double)MB*(double)C[0]*(double)C[1]*(double)iters) / (1000.0*1000.0*1000.0);
    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);
    printf("PERFDUMP,BP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
    for ( i = 0; i < num_layers; ++i ) {
      printf("%i,", C[i] );
    }
    printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
#endif
  }

  if (type == 'A') {
    printf("##########################################\n");
    printf("# Performance - FWD-BWD (custom-Storage) #\n");
    printf("##########################################\n");

    /* Timers: */
    fwd_time = (unsigned long long *) malloc(sizeof(unsigned long long) * nThreads);
    bwd_time = (unsigned long long *) malloc(sizeof(unsigned long long) * nThreads);
    solver_time = (unsigned long long *) malloc(sizeof(unsigned long long) * nThreads);

    /* Calculate chunks of weights used on each nume node on FWD based on FWD thread decomposition */
    setup_my_numa_fwd(&numa_thr_cfg, num_layers, my_fc_fwd);
    /* Calculate chunks of weights used on each nume node on BWD/d based on BWD/d thread decomposition */
    setup_my_numa_bwd_d(&numa_thr_cfg, num_layers, my_fc_bwd);
    /* NUMA aware allocations of buffers needed for FWD */
    allocate_numa_buffers_fwd(&numa_thr_cfg, num_layers, my_fc_fwd);
    /* NUMA aware allocations of buffers needed for BWD */
    allocate_numa_buffers_bwd_d(&numa_thr_cfg, num_layers, my_fc_bwd);

    /* Utility needed for transpoisition of weigths on BWD/d: get numa node based on current ofm */
    int **fwd_ofm_to_node = (int**)malloc(sizeof(int*) * num_layers);
    set_fwd_ofm_to_node(fwd_ofm_to_node, &numa_thr_cfg, num_layers, my_fc_fwd);

    l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#   pragma omp parallel private(i,j)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      fwd_time[tid] = 0;
      bwd_time[tid] = 0;
      solver_time[tid] = 0;
      const int numa_node = numa_node_of_cpu(tid);
      for ( i = 0; i < num_layers; ++i) {
        /* Copy original weights to NUMA FWD buffers. Threading decomposition is the same with FWD. */
        copy_to_numa_buffers_fwd(&numa_thr_cfg[numa_node], my_fc_fwd[i], fil_libxsmm[i], numa_node, i, tid, 0);
      }
      for (j = 0; j < iters; ++j) {
       unsigned long long fwd_time_start = libxsmm_timer_tick();
       for ( i = 0; i < num_layers; ++i) {
          /* FWD: Use weights from NUMA FWD buffers */
          my_fc_fwd_exec( my_fc_fwd[i], act_libxsmm[i], act_libxsmm[i+1],
                          bias_libxsmm[i], relumask_libxsmm[i], 0, tid, scratch, &numa_thr_cfg[numa_node], i );
        }
       fwd_time[tid] += (libxsmm_timer_tick() - fwd_time_start);
#ifdef USE_SOFTMAX
        my_smax_fwd_exec( my_smax_fwd, act_libxsmm[num_layers], act_libxsmm[num_layers+1], label_libxsmm, &loss,
                          0, tid, scratch );
        my_smax_bwd_exec( my_smax_bwd, delact_libxsmm[num_layers], act_libxsmm[num_layers+1], label_libxsmm,
                          0, tid, scratch );
#endif
        for ( i = num_layers-1; i > 0; --i) {
          unsigned long long bwd_time_start = libxsmm_timer_tick();
          /* Transpose weights from NUMA FWD buffers to NUMA BWD buffer. Threading decomposition is the same with BWD/d. */
          my_fc_bwd_d_transpose( my_fc_bwd[i], tid , &numa_thr_cfg, numa_node, i, fwd_ofm_to_node[i] );
          /* BWD/d: Use weights from NUMA BWD buffers */
          my_fc_bwd_exec( my_fc_bwd[i], delact_libxsmm[i], delact_libxsmm[i+1], delfil_libxsmm[i],
                          act_libxsmm[i], delbias_libxsmm[i], relumask_libxsmm[i], MY_PASS_BWD, 0, tid, scratch, &numa_thr_cfg[numa_node], i );
          bwd_time[tid] += (libxsmm_timer_tick() - bwd_time_start);
          /* Solver: Update NUMA FWD buffers. Threading decomposition is the same with FWD. */
          unsigned long long solver_time_start = libxsmm_timer_tick();
          my_opt_exec( my_opt[i], delfil_libxsmm[i], 0, tid, &numa_thr_cfg[numa_node], i, my_fc_fwd[i] );
          solver_time[tid] += (libxsmm_timer_tick() - solver_time_start);

        }
        /* BWD/w: todo */
        unsigned long long bwd_time_start = libxsmm_timer_tick();
        my_fc_bwd_exec( my_fc_bwd[0], delact_libxsmm[0], delact_libxsmm[0+1], delfil_libxsmm[0],
                        act_libxsmm[0], delbias_libxsmm[0], relumask_libxsmm[0], MY_PASS_BWD_W, 0, tid, scratch, &numa_thr_cfg[numa_node], 0 );
        bwd_time[tid] += (libxsmm_timer_tick() - bwd_time_start);
        /* Solver: Update NUMA FWD buffers. Threading decomposition is the same with FWD. */
        unsigned long long solver_time_start = libxsmm_timer_tick();
        my_opt_exec( my_opt[0], delfil_libxsmm[0], 0, tid, &numa_thr_cfg[numa_node], 0, my_fc_fwd[0] );
        solver_time[tid] += (libxsmm_timer_tick() - solver_time_start);
      }
      /* Copy result from NUMA FWD Buffers to original weights. Threading decomposition is the same with FWD. */
      for ( i = 0; i < num_layers; ++i) {
        copy_to_numa_buffers_fwd(&numa_thr_cfg[numa_node], my_fc_fwd[i], fil_libxsmm[i], numa_node, i, tid, 1);
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    free_fwd_ofm_to_node(fwd_ofm_to_node, num_layers);
    free(fwd_ofm_to_node);
#ifdef CHECK_L1
#if 1
    /* Print some norms on last act for fwd and weights of first layer after all iterations */
    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, MB*C[num_layers], 1, act_libxsmm[num_layers], act_libxsmm[num_layers], 0, 0);
    printf("L1 of act[num_layers]  : %.25g\n", norms_fwd.l1_ref);
    libxsmm_matdiff_reduce(&diff, &norms_fwd);
    libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, C[0]*C[1], 1, fil_libxsmm[0], fil_libxsmm[0], 0, 0);
    printf("L1 of wt[0]  : %.25g\n", norms_bwd.l1_ref);
    libxsmm_matdiff_reduce(&diff, &norms_bwd);
#else
    {
      int e = 0;
      FILE *fileAct, *fileWt;
      fileAct = fopen("acts.txt","w+");
      if (fileAct != NULL) {
        for (e = 0; e < MB*C[num_layers]; e++) {
          fprintf(fileAct, "%.10g\n", *((float*)act_libxsmm[num_layers] + e));
        }
        fclose(fileAct);
      }
      fileWt = fopen("weights.txt","w+");
      if (fileWt != NULL) {
        for (e = 0; e < C[0]*C[1]; e++) {
          fprintf(fileWt, "%.10g\n", *((float*)fil_libxsmm[0] + e));
        }
        fclose(fileWt);
      }
    }
#endif
#endif

    gflop = 0.0;
    for ( i = num_layers-1; i > 0; --i) {
      gflop += (6.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    gflop += (4.0*(double)MB*(double)C[0]*(double)C[1]*(double)iters) / (1000.0*1000.0*1000.0);
    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);
    printf("PERFDUMP,BP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
    for ( i = 0; i < num_layers; ++i ) {
      printf("%i,", C[i] );
    }
    printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
    unsigned long long max_fwd_time = 0, max_bwd_time = 0, max_solver_time = 0;
    for (i = 0; i < nThreads; i++) {
        if (max_fwd_time < fwd_time[i]) max_fwd_time = fwd_time[i];
        if (max_bwd_time < bwd_time[i]) max_bwd_time = bwd_time[i];
        if (max_solver_time < solver_time[i]) max_solver_time = solver_time[i];
    }
    printf("Profiling: fwd_time = %lld, bwd_time = %lld, solver_time = %lld\n",
        max_fwd_time, max_bwd_time, max_solver_time);
  }

  /* deallocate data */
  if ( scratch != NULL ) {
    libxsmm_free(scratch);
  }
  for ( i = 0; i < num_layers; ++i ) {
    if ( i == 0 ) {
#ifdef ACT_NUMA_INTERLEAVED
      numa_free(act_libxsmm[i], MB*C[i]*sizeof(float));
#else
      libxsmm_free(act_libxsmm[i]);
#endif
      libxsmm_free(delact_libxsmm[i]);
    }
#ifdef ACT_NUMA_INTERLEAVED
    numa_free(act_libxsmm[i+1], MB*C[i+1]*sizeof(float));
#else
    libxsmm_free(act_libxsmm[i+1]);
#endif
    libxsmm_free(delact_libxsmm[i+1]);

    libxsmm_free(fil_libxsmm[i]);
    libxsmm_free(delfil_libxsmm[i]);
    libxsmm_free(bias_libxsmm[i]);
    libxsmm_free(delbias_libxsmm[i]);
    libxsmm_free(relumask_libxsmm[i]);
  }
#ifdef ACT_NUMA_INTERLEAVED
  numa_free(act_libxsmm[num_layers+1], MB*C[num_layers+1]*sizeof(float));
#else
  libxsmm_free(act_libxsmm[num_layers+1]);
#endif
  libxsmm_free(label_libxsmm);

  for (i = 0; i < numa_num_configured_nodes(); i++) {
    free(numa_thr_cfg[i].blocksOFm_s);
    free(numa_thr_cfg[i].blocksOFm_e);
    free(numa_thr_cfg[i].blocksIFm_tr_s);
    free(numa_thr_cfg[i].blocksIFm_tr_e);
    for (j = 0; j < num_layers; j++) {
      numa_free_aligned(numa_thr_cfg[i].scratch[j], numa_thr_cfg[i].layer_size[j]);
    }
    free(numa_thr_cfg[i].scratch);
    free(numa_thr_cfg[i].layer_size);
    numa_free_aligned(numa_thr_cfg[i].bwd_d_scratch, numa_thr_cfg[i].bwd_d_scratch_size);
  }
  free(numa_thr_cfg);

  free( my_opt );
  free( my_fc_fwd );
  free( my_fc_bwd );

  free( act_libxsmm );
  free( delact_libxsmm );
  free( fil_libxsmm );
  free( delfil_libxsmm );
  free( bias_libxsmm );
  free( delbias_libxsmm );
  free( relumask_libxsmm );

  free( C );

  /* some empty lines at the end */
  printf("\n\n\n");

  return 0;
}

