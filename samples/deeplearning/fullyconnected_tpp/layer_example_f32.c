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
  libxsmm_blasint fwd_column_teams;
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
  libxsmm_blasint bwd_row_teams;
  libxsmm_blasint bwd_column_teams;
  libxsmm_blasint upd_bf;
  libxsmm_blasint upd_2d_blocking;
  libxsmm_blasint upd_row_teams;
  libxsmm_blasint upd_column_teams;
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
  /* Transpose kernel used for weight transpose in bwd pass */
  res.norm_to_normT_kernel = libxsmm_dispatch_meltw_unary(bk, bc, &ldaT, &lda, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
  if ( res.norm_to_normT_kernel == NULL ) {
    fprintf( stderr, "JIT for transpose TPP tr_kernel failed. Bailing...!\n");
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
  libxsmm_blasint mb1ofm1 = 0, mb1 = 0, ofm1 = 0, ifm1 = 0, mb2 = 0, ofm2 = 0;
  libxsmm_blasint im_tasks_per_thread = 0, in_tasks_per_thread = 0;
  libxsmm_blasint my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0;
  libxsmm_blasint my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;

  LIBXSMM_VLA_DECL(4, float,           output, out_act_ptr, nBlocksOFm, cfg.bn, cfg.bk);
  LIBXSMM_VLA_DECL(4, const float,      input,  in_act_ptr, nBlocksIFm, cfg.bn, cfg.bc);
  LIBXSMM_VLA_DECL(4, const float,     filter,      wt_ptr, nBlocksIFm, cfg.bc, cfg.bk);
  LIBXSMM_VLA_DECL(2, const float,       bias,    bias_ptr,                     cfg.bk);
  LIBXSMM_VLA_DECL(4, unsigned char, relumask,    relu_ptr, nBlocksOFm, cfg.bn, cfg.bk);

  unsigned long long  blocks = nBlocksIFm;
  libxsmm_blasint CB_BLOCKS = nBlocksIFm, BF = 1;
  LIBXSMM_UNUSED( scratch );

  BF = cfg.fwd_bf;
  CB_BLOCKS = nBlocksIFm/BF;
  blocks = CB_BLOCKS;

  row_teams = cfg.fwd_row_teams;
  column_teams = cfg.fwd_column_teams;
  my_col_id = ltid % column_teams;
  my_row_id = ltid / column_teams;
  im_tasks_per_thread = LIBXSMM_UPDIV(nBlocksMB, row_teams);
  in_tasks_per_thread = LIBXSMM_UPDIV(nBlocksOFm, column_teams);
  my_im_start = LIBXSMM_MIN(my_row_id * im_tasks_per_thread, nBlocksMB);
  my_im_end = LIBXSMM_MIN((my_row_id+1) * im_tasks_per_thread, nBlocksMB);
  my_in_start = LIBXSMM_MIN(my_col_id * in_tasks_per_thread, nBlocksOFm);
  my_in_end = LIBXSMM_MIN((my_col_id+1) * in_tasks_per_thread, nBlocksOFm);

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  if (cfg.fwd_2d_blocking == 1) {
    if (BF > 1) {
      for (ifm1 = 0; ifm1 < BF; ++ifm1) {
        for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
          for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
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
      for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
        for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
          if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
            for ( mb2 = 0; mb2 < cfg.bn; ++mb2 ) {
              for ( ofm2 = 0; ofm2 < cfg.bk; ++ofm2 ) {
                LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, cfg.bn, cfg.bk) = LIBXSMM_VLA_ACCESS(2, bias, ofm1, ofm2, cfg.bk);
              }
            }
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
  libxsmm_meltw_unary_param trans_param;

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

  const float *grad_output_ptr = (const float*)(((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) ? ((float*)scratch)+(cfg.C*cfg.K) : dout_act_ptr);
  LIBXSMM_VLA_DECL(4, const float, doutput_orig,    dout_act_ptr, nBlocksOFm, bn, bk);
  LIBXSMM_VLA_DECL(4,       float,      doutput, grad_output_ptr, nBlocksOFm, bn, bk);

  LIBXSMM_VLA_DECL(2,         float,    dbias, dbias_ptr,                     cfg.bk);
  LIBXSMM_VLA_DECL(4, unsigned char, relumask,  relu_ptr, nBlocksOFm, cfg.bn, cfg.bk);

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

    LIBXSMM_VLA_DECL(4, const float,    filter,          wt_ptr, nBlocksIFm, bc, bk);
    LIBXSMM_VLA_DECL(4,       float,    dinput,     din_act_ptr, nBlocksIFm, bn, bc);
    LIBXSMM_VLA_DECL(4,       float, filter_tr, (float*)scratch, nBlocksOFm, bk, bc);

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
      im_tasks_per_thread = LIBXSMM_UPDIV(nBlocksMB, row_teams);
      in_tasks_per_thread = LIBXSMM_UPDIV(nBlocksIFm, column_teams);
      my_im_start = LIBXSMM_MIN(my_row_id * im_tasks_per_thread, nBlocksMB);
      my_im_end = LIBXSMM_MIN((my_row_id+1) * im_tasks_per_thread, nBlocksMB);
      my_in_start = LIBXSMM_MIN(my_col_id * in_tasks_per_thread, nBlocksIFm);
      my_in_end = LIBXSMM_MIN((my_col_id+1) * in_tasks_per_thread, nBlocksIFm);
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
          for (ifm1 = my_in_start; ifm1 < my_in_end; ++ifm1) {
            for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
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
        for (ifm1 = my_in_start; ifm1 < my_in_end; ++ifm1) {
          for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
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
    libxsmm_blasint im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;

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
      row_teams = cfg.upd_row_teams;
      column_teams = cfg.upd_column_teams;
      my_col_id = ltid % column_teams;
      my_row_id = ltid / column_teams;
      im_tasks_per_thread = LIBXSMM_UPDIV(nBlocksIFm, row_teams);
      in_tasks_per_thread = LIBXSMM_UPDIV(nBlocksOFm, column_teams);
      my_im_start = LIBXSMM_MIN(my_row_id * im_tasks_per_thread, nBlocksIFm);
      my_im_end = LIBXSMM_MIN((my_row_id+1) * im_tasks_per_thread, nBlocksIFm);
      my_in_start = LIBXSMM_MIN(my_col_id * in_tasks_per_thread, nBlocksOFm);
      my_in_end = LIBXSMM_MIN((my_col_id+1) * in_tasks_per_thread, nBlocksOFm);
    }

    if (use_2d_blocking == 1) {
      if (BF == 1) {
        for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
          for (ifm1 = my_im_start; ifm1 < my_im_end; ++ifm1) {
            cfg.gemm_upd2(&LIBXSMM_VLA_ACCESS(4, doutput, 0, ofm1, 0, 0, nBlocksOFm, bn, bk),
                          &LIBXSMM_VLA_ACCESS(4, input,   0, ifm1, 0, 0, nBlocksIFm, bn, bc),
                          &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk), &blocks);
          }
        }
      } else {
        for (bfn = 0; bfn < BF; bfn++) {
          for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
            for (ifm1 = my_im_start; ifm1 < my_im_end; ++ifm1) {
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
  relumask_libxsmm           = (unsigned char*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(unsigned char), 2097152);

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

